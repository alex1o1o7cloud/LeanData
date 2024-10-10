import Mathlib

namespace greatest_power_under_500_l3528_352850

theorem greatest_power_under_500 (a b : ℕ) :
  a > 0 → b > 1 → a^b < 500 →
  (∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x^y ≤ a^b) →
  a + b = 24 := by
sorry

end greatest_power_under_500_l3528_352850


namespace composition_f_one_ninth_l3528_352857

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem composition_f_one_ninth :
  f (f (1/9)) = 1/4 := by sorry

end composition_f_one_ninth_l3528_352857


namespace ellipse_intersection_theorem_l3528_352897

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- A line with a given slope -/
structure Line where
  slope : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => p.1^2 / (a^2) + p.2^2 / (b^2) = 1

/-- The equation of a line -/
def line_equation (m b : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => p.2 = m * p.1 + b

theorem ellipse_intersection_theorem (C : Ellipse) (l : Line) :
  C.center = (0, 0) ∧ C.left_focus = (-Real.sqrt 3, 0) ∧ C.right_vertex = (2, 0) ∧ l.slope = 1/2 →
  (∃ a b : ℝ, standard_equation a b = standard_equation 2 1) ∧
  (∃ chord_length : ℝ, chord_length ≤ Real.sqrt 10 ∧
    ∀ other_length : ℝ, other_length ≤ chord_length) ∧
  (∃ b : ℝ, line_equation (1/2) b = line_equation (1/2) 0 →
    ∀ other_b : ℝ, ∃ length : ℝ, length ≤ Real.sqrt 10) :=
by sorry

end ellipse_intersection_theorem_l3528_352897


namespace max_k_value_l3528_352852

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ 1 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 = 1^2 * ((x^2 / y^2) + (y^2 / x^2)) + 1 * ((x / y) + (y / x)) :=
sorry

end max_k_value_l3528_352852


namespace set_operations_l3528_352837

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | (x - 1) / (x - 6) < 0}

theorem set_operations :
  (A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 6}) ∧
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8}) ∧
  ((Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x < 2}) := by sorry

end set_operations_l3528_352837


namespace basketball_win_rate_l3528_352827

theorem basketball_win_rate (total_games : ℕ) (first_part_games : ℕ) (first_part_wins : ℕ) 
  (remaining_games : ℕ) (target_percentage : ℚ) :
  total_games = first_part_games + remaining_games →
  (first_part_wins : ℚ) / (first_part_games : ℚ) > target_percentage →
  ∃ (remaining_wins : ℕ), 
    remaining_wins ≤ remaining_games ∧ 
    ((first_part_wins + remaining_wins : ℚ) / (total_games : ℚ) ≥ target_percentage) ∧
    (∀ (x : ℕ), x < remaining_wins → 
      (first_part_wins + x : ℚ) / (total_games : ℚ) < target_percentage) :=
by
  sorry

-- Example usage
example : 
  ∃ (remaining_wins : ℕ), 
    remaining_wins ≤ 35 ∧ 
    ((45 + remaining_wins : ℚ) / 90 ≥ 3/4) ∧
    (∀ (x : ℕ), x < remaining_wins → (45 + x : ℚ) / 90 < 3/4) :=
basketball_win_rate 90 55 45 35 (3/4)
  (by norm_num)
  (by norm_num)

end basketball_win_rate_l3528_352827


namespace smallest_positive_a_for_two_roots_in_unit_interval_l3528_352822

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate to check if a quadratic function has two distinct roots in (0,1) -/
def has_two_distinct_roots_in_unit_interval (f : QuadraticFunction) : Prop :=
  ∃ (r s : ℝ), 0 < r ∧ r < 1 ∧ 0 < s ∧ s < 1 ∧ r ≠ s ∧
    f.a * r^2 + f.b * r + f.c = 0 ∧
    f.a * s^2 + f.b * s + f.c = 0

/-- The main theorem stating the smallest positive integer a -/
theorem smallest_positive_a_for_two_roots_in_unit_interval :
  ∃ (a : ℤ), a > 0 ∧
    (∀ (f : QuadraticFunction), f.a = a → has_two_distinct_roots_in_unit_interval f) ∧
    (∀ (a' : ℤ), 0 < a' → a' < a →
      ∃ (f : QuadraticFunction), f.a = a' ∧ ¬has_two_distinct_roots_in_unit_interval f) ∧
    a = 5 :=
  sorry

end smallest_positive_a_for_two_roots_in_unit_interval_l3528_352822


namespace lamp_position_probability_l3528_352819

/-- The probability that a randomly chosen point on a line segment of length 6
    is at least 2 units away from both endpoints is 1/3. -/
theorem lamp_position_probability : 
  let total_length : ℝ := 6
  let min_distance : ℝ := 2
  let favorable_length : ℝ := total_length - 2 * min_distance
  favorable_length / total_length = 1 / 3 := by sorry

end lamp_position_probability_l3528_352819


namespace moneybox_fills_in_60_weeks_l3528_352818

/-- The number of weeks it takes for Monica's moneybox to get full -/
def weeks_to_fill : ℕ := sorry

/-- The amount Monica puts into her moneybox each week -/
def weekly_savings : ℕ := 15

/-- The number of times Monica repeats the saving process -/
def repetitions : ℕ := 5

/-- The total amount Monica takes to the bank -/
def total_savings : ℕ := 4500

/-- Theorem stating that the moneybox gets full in 60 weeks -/
theorem moneybox_fills_in_60_weeks : 
  weeks_to_fill = 60 ∧ 
  weekly_savings * weeks_to_fill * repetitions = total_savings :=
sorry

end moneybox_fills_in_60_weeks_l3528_352818


namespace coupon_savings_difference_l3528_352867

/-- Represents the savings from Coupon A (15% off the listed price) -/
def savingsA (price : ℝ) : ℝ := 0.15 * price

/-- Represents the savings from Coupon B ($30 flat discount) -/
def savingsB : ℝ := 30

/-- Represents the savings from Coupon C (20% off the amount over $150) -/
def savingsC (price : ℝ) : ℝ := 0.20 * (price - 150)

/-- The theorem to be proved -/
theorem coupon_savings_difference : 
  ∃ (min_price max_price : ℝ),
    (∀ price, price > 150 → 
      (savingsA price > savingsB ∧ savingsA price > savingsC price) ↔ 
      (min_price ≤ price ∧ price ≤ max_price)) ∧
    max_price - min_price = 400 :=
sorry

end coupon_savings_difference_l3528_352867


namespace prime_equation_solutions_l3528_352810

theorem prime_equation_solutions (p : ℕ) :
  (Prime p ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 := by
  sorry

end prime_equation_solutions_l3528_352810


namespace triangle_side_equations_l3528_352858

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude and median
def altitude_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0
def median_equation (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Define the equations of the sides
def side_AB_equation (x y : ℝ) : Prop := 2*x - y + 1 = 0
def side_BC_equation (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def side_AC_equation (x y : ℝ) : Prop := y = 1

theorem triangle_side_equations 
  (tri : Triangle)
  (h1 : tri.A = (0, 1))
  (h2 : ∀ x y, altitude_equation x y → (x - tri.A.1) * (tri.B.2 - tri.A.2) = -(y - tri.A.2) * (tri.B.1 - tri.A.1))
  (h3 : ∀ x y, median_equation x y → 2 * x = tri.A.1 + tri.C.1 ∧ 2 * y = tri.A.2 + tri.C.2) :
  (∀ x y, side_AB_equation x y ↔ (y - tri.A.2) = ((tri.B.2 - tri.A.2) / (tri.B.1 - tri.A.1)) * (x - tri.A.1)) ∧
  (∀ x y, side_BC_equation x y ↔ (y - tri.B.2) = ((tri.C.2 - tri.B.2) / (tri.C.1 - tri.B.1)) * (x - tri.B.1)) ∧
  (∀ x y, side_AC_equation x y ↔ (y - tri.A.2) = ((tri.C.2 - tri.A.2) / (tri.C.1 - tri.A.1)) * (x - tri.A.1)) :=
by sorry

end triangle_side_equations_l3528_352858


namespace meet_once_l3528_352887

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def count_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : Movement where
  michael_speed := 4
  truck_speed := 12
  pail_distance := 200
  truck_stop_time := 20

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once :
  count_meetings problem_scenario = 1 :=
sorry

end meet_once_l3528_352887


namespace constant_calculation_l3528_352815

theorem constant_calculation (n : ℤ) (c : ℝ) : 
  (∀ k : ℤ, c * k^2 ≤ 8100) → (∀ m : ℤ, m ≤ 8) → c = 126.5625 := by
sorry

end constant_calculation_l3528_352815


namespace square_side_length_l3528_352814

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 8 →
  rectangle_width = 10 →
  4 * square_side = 2 * (rectangle_length + rectangle_width) →
  square_side = 9 := by
sorry

end square_side_length_l3528_352814


namespace alex_jimmy_yellow_ratio_l3528_352829

-- Define the number of marbles each person has
def lorin_black : ℕ := 4
def jimmy_yellow : ℕ := 22
def alex_total : ℕ := 19

-- Define Alex's black marbles as twice Lorin's
def alex_black : ℕ := 2 * lorin_black

-- Define Alex's yellow marbles
def alex_yellow : ℕ := alex_total - alex_black

-- Theorem to prove
theorem alex_jimmy_yellow_ratio :
  (alex_yellow : ℚ) / jimmy_yellow = 1 / 2 := by
  sorry

end alex_jimmy_yellow_ratio_l3528_352829


namespace cubic_equation_roots_l3528_352807

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 10*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 37 := by
sorry

end cubic_equation_roots_l3528_352807


namespace taxi_fare_equation_l3528_352826

/-- Taxi fare calculation -/
theorem taxi_fare_equation 
  (x : ℝ) 
  (h_distance : x > 3) 
  (starting_price : ℝ := 6) 
  (price_per_km : ℝ := 2.4) 
  (total_fare : ℝ := 13.2) :
  starting_price + price_per_km * (x - 3) = total_fare := by
  sorry

end taxi_fare_equation_l3528_352826


namespace common_root_pairs_l3528_352889

theorem common_root_pairs (n : ℕ) (hn : n > 1) :
  ∀ s t : ℤ, (∃ x : ℝ, x^n + s*x = 2007 ∧ x^n + t*x = 2008) ↔ 
  ((s = 2006 ∧ t = 2007) ∨ 
   (s = -2008 ∧ t = -2009 ∧ Even n) ∨ 
   (s = -2006 ∧ t = -2007 ∧ Odd n)) :=
by sorry

end common_root_pairs_l3528_352889


namespace unique_matching_number_l3528_352841

/-- A function that checks if two numbers match in exactly one digit position -/
def match_one_digit (a b : ℕ) : Prop :=
  ∃! i : Fin 3, (a / 10^i.val % 10) = (b / 10^i.val % 10)

/-- The theorem stating that 729 is the only three-digit number matching one digit with each guess -/
theorem unique_matching_number : 
  ∀ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    match_one_digit n 109 ∧ 
    match_one_digit n 704 ∧ 
    match_one_digit n 124 
    → n = 729 := by
  sorry


end unique_matching_number_l3528_352841


namespace hat_shoppe_pricing_l3528_352843

theorem hat_shoppe_pricing (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) = 0.975 * x := by sorry

end hat_shoppe_pricing_l3528_352843


namespace hexagon_angle_measure_l3528_352865

theorem hexagon_angle_measure :
  ∀ (a b c d e : ℝ),
    a = 135 ∧ b = 150 ∧ c = 120 ∧ d = 130 ∧ e = 100 →
    ∃ (q : ℝ),
      q = 85 ∧
      a + b + c + d + e + q = 720 :=
by sorry

end hexagon_angle_measure_l3528_352865


namespace inscribed_circle_radius_l3528_352871

/-- A sector OAB is a third of a circle with radius 6 cm. 
    An inscribed circle is tangent to the sector at three points. -/
def sector_with_inscribed_circle (r : ℝ) : Prop :=
  r > 0 ∧ 
  ∃ (R : ℝ), R = 6 ∧
  ∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧
  x = R * Real.sin θ ∧
  y = R * (1 - Real.cos θ)

/-- The radius of the inscribed circle in the sector described above is 6√2 - 6 cm. -/
theorem inscribed_circle_radius :
  ∀ r : ℝ, sector_with_inscribed_circle r → r = 6 * (Real.sqrt 2 - 1) :=
by sorry

end inscribed_circle_radius_l3528_352871


namespace parallelogram_xy_sum_l3528_352882

/-- A parallelogram with sides a, b, c, d where opposite sides are equal -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  opposite_sides_equal : a = c ∧ b = d

/-- The specific parallelogram from the problem -/
def problem_parallelogram (x y : ℝ) : Parallelogram where
  a := 6 * y - 2
  b := 12
  c := 3 * x + 4
  d := 9
  opposite_sides_equal := by sorry

theorem parallelogram_xy_sum (x y : ℝ) :
  (problem_parallelogram x y).a = (problem_parallelogram x y).c ∧
  (problem_parallelogram x y).b = (problem_parallelogram x y).d →
  x + y = 4 := by sorry

end parallelogram_xy_sum_l3528_352882


namespace xyz_values_l3528_352879

theorem xyz_values (x y z : ℝ) 
  (eq1 : x * y - 5 * y = 20)
  (eq2 : y * z - 5 * z = 20)
  (eq3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := by
sorry

end xyz_values_l3528_352879


namespace sapling_growth_relation_l3528_352880

/-- Represents the height of a sapling over time -/
def sapling_height (x : ℝ) : ℝ :=
  50 * x + 100

theorem sapling_growth_relation (x : ℝ) (y : ℝ) 
  (h1 : sapling_height 0 = 100) 
  (h2 : ∀ x1 x2, sapling_height x2 - sapling_height x1 = 50 * (x2 - x1)) :
  y = sapling_height x :=
by sorry

end sapling_growth_relation_l3528_352880


namespace multiplier_is_five_l3528_352848

/-- Given a number that equals some times the difference between itself and 4,
    prove that when the number is 5, the multiplier is also 5. -/
theorem multiplier_is_five (n m : ℝ) : n = m * (n - 4) → n = 5 → m = 5 := by
  sorry

end multiplier_is_five_l3528_352848


namespace soda_production_in_8_hours_l3528_352835

/-- Represents the production rate of a soda machine -/
structure SodaMachine where
  cans_per_interval : ℕ
  interval_minutes : ℕ

/-- Calculates the number of cans produced in a given number of hours -/
def cans_produced (machine : SodaMachine) (hours : ℕ) : ℕ :=
  let intervals_per_hour : ℕ := 60 / machine.interval_minutes
  let total_intervals : ℕ := hours * intervals_per_hour
  machine.cans_per_interval * total_intervals

theorem soda_production_in_8_hours (machine : SodaMachine)
    (h1 : machine.cans_per_interval = 30)
    (h2 : machine.interval_minutes = 30) :
    cans_produced machine 8 = 480 := by
  sorry

end soda_production_in_8_hours_l3528_352835


namespace duck_purchase_difference_l3528_352847

/-- Represents the number of ducks bought by each person -/
structure DuckPurchase where
  adelaide : ℕ
  ephraim : ℕ
  kolton : ℕ

/-- The conditions of the duck purchase problem -/
def DuckProblemConditions (d : DuckPurchase) : Prop :=
  d.adelaide = 2 * d.ephraim ∧
  d.adelaide = 30 ∧
  (d.adelaide + d.ephraim + d.kolton) / 3 = 35

/-- The theorem stating the difference between Kolton's and Ephraim's duck purchases -/
theorem duck_purchase_difference (d : DuckPurchase) :
  DuckProblemConditions d → d.kolton - d.ephraim = 45 := by
  sorry


end duck_purchase_difference_l3528_352847


namespace circle_division_l3528_352851

theorem circle_division (OA : ℝ) (OA_pos : OA > 0) :
  ∃ (OC OB : ℝ),
    OC = (OA * Real.sqrt 3) / 3 ∧
    OB = (OA * Real.sqrt 6) / 3 ∧
    π * OC^2 = π * (OB^2 - OC^2) ∧
    π * (OB^2 - OC^2) = π * (OA^2 - OB^2) :=
by sorry

end circle_division_l3528_352851


namespace sea_hidden_by_cloud_l3528_352888

theorem sea_hidden_by_cloud (total_landscape visible_island cloud_cover : ℚ) :
  cloud_cover = 1/2 ∧ 
  visible_island = 1/4 ∧ 
  visible_island = 3/4 * (visible_island + (cloud_cover - 1/2)) →
  cloud_cover - (cloud_cover - 1/2) - visible_island = 5/12 :=
by sorry

end sea_hidden_by_cloud_l3528_352888


namespace credits_to_graduate_l3528_352820

/-- The number of semesters in college -/
def semesters : ℕ := 8

/-- The number of classes taken per semester -/
def classes_per_semester : ℕ := 5

/-- The number of credits per class -/
def credits_per_class : ℕ := 3

/-- The total number of credits needed to graduate -/
def total_credits : ℕ := semesters * classes_per_semester * credits_per_class

theorem credits_to_graduate : total_credits = 120 := by
  sorry

end credits_to_graduate_l3528_352820


namespace N2O_molecular_weight_l3528_352845

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of N2O in g/mol -/
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

/-- Theorem stating that the molecular weight of N2O is 44.02 g/mol -/
theorem N2O_molecular_weight : molecular_weight_N2O = 44.02 := by
  sorry

end N2O_molecular_weight_l3528_352845


namespace waiter_earnings_l3528_352894

def lunch_shift (total_customers : ℕ) (tipping_customers : ℕ) 
  (tip_8 : ℕ) (tip_10 : ℕ) (tip_12 : ℕ) (meal_cost : ℕ) : ℕ :=
  let total_tips := 8 * tip_8 + 10 * tip_10 + 12 * tip_12
  total_tips - meal_cost

theorem waiter_earnings : 
  lunch_shift 12 6 3 2 1 5 = 51 := by sorry

end waiter_earnings_l3528_352894


namespace inverse_variation_problem_l3528_352839

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_problem (a b : ℝ → ℝ) 
  (h1 : VaryInversely a b) 
  (h2 : a 1 = 1500) 
  (h3 : b 1 = 0.25) 
  (h4 : a 2 = 3000) : 
  b 2 = 0.125 := by
sorry


end inverse_variation_problem_l3528_352839


namespace inequality_property_l3528_352895

theorem inequality_property (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_property_l3528_352895


namespace circle_line_distance_l3528_352893

theorem circle_line_distance (a : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 4)^2 = 4}
  let line : Set (ℝ × ℝ) := {p | a * p.1 + p.2 - 1 = 0}
  let center : ℝ × ℝ := (1, 4)
  (∀ p ∈ line, ((p.1 - center.1)^2 + (p.2 - center.2)^2).sqrt ≥ 1) ∧
  (∃ p ∈ line, ((p.1 - center.1)^2 + (p.2 - center.2)^2).sqrt = 1) →
  a = -4/3 := by
sorry


end circle_line_distance_l3528_352893


namespace polynomial_factorization_l3528_352828

theorem polynomial_factorization (a b : ℝ) :
  (a^2 + 10*a + 25) - b^2 = (a + 5 + b) * (a + 5 - b) := by
  sorry

#check polynomial_factorization

end polynomial_factorization_l3528_352828


namespace car_dealership_monthly_payment_l3528_352878

/-- Calculates the total monthly payment for employees in a car dealership --/
theorem car_dealership_monthly_payment 
  (fiona_hours : ℕ) 
  (john_hours : ℕ) 
  (jeremy_hours : ℕ) 
  (hourly_rate : ℕ) 
  (h1 : fiona_hours = 40)
  (h2 : john_hours = 30)
  (h3 : jeremy_hours = 25)
  (h4 : hourly_rate = 20)
  : (fiona_hours + john_hours + jeremy_hours) * hourly_rate * 4 = 7600 := by
  sorry


end car_dealership_monthly_payment_l3528_352878


namespace max_people_served_l3528_352846

theorem max_people_served (total_budget : ℚ) (min_food_spend : ℚ) (cheapest_food_cost : ℚ) (cheapest_drink_cost : ℚ) 
  (h1 : total_budget = 12.5)
  (h2 : min_food_spend = 10)
  (h3 : cheapest_food_cost = 0.6)
  (h4 : cheapest_drink_cost = 0.5) :
  ∃ (n : ℕ), n = 10 ∧ 
    n * (cheapest_food_cost + cheapest_drink_cost) ≤ total_budget ∧
    n * cheapest_food_cost ≥ min_food_spend ∧
    ∀ (m : ℕ), m > n → 
      m * (cheapest_food_cost + cheapest_drink_cost) > total_budget ∨
      m * cheapest_food_cost < min_food_spend :=
by
  sorry

#check max_people_served

end max_people_served_l3528_352846


namespace square_root_range_l3528_352866

theorem square_root_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end square_root_range_l3528_352866


namespace growth_percentage_calculation_l3528_352890

def previous_height : Real := 139.65
def current_height : Real := 147.0

theorem growth_percentage_calculation :
  let difference := current_height - previous_height
  let growth_rate := difference / previous_height
  let growth_percentage := growth_rate * 100
  ∃ ε > 0, abs (growth_percentage - 5.26) < ε :=
sorry

end growth_percentage_calculation_l3528_352890


namespace max_sum_product_sqrt_max_value_quarter_equality_condition_l3528_352824

theorem max_sum_product_sqrt (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  ∀ y1 y2 y3 y4 : ℝ, 
    y1 ≥ 0 → y2 ≥ 0 → y3 ≥ 0 → y4 ≥ 0 → 
    y1 + y2 + y3 + y4 = 1 → 
    sum_prod ≥ (y1 + y2) * Real.sqrt (y1 * y2) + 
               (y1 + y3) * Real.sqrt (y1 * y3) + 
               (y1 + y4) * Real.sqrt (y1 * y4) + 
               (y2 + y3) * Real.sqrt (y2 * y3) + 
               (y2 + y4) * Real.sqrt (y2 * y4) + 
               (y3 + y4) * Real.sqrt (y3 * y4) :=
by sorry

theorem max_value_quarter (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  sum_prod ≤ 3/4 :=
by sorry

theorem equality_condition (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  sum_prod = 3/4 ↔ x1 = 1/4 ∧ x2 = 1/4 ∧ x3 = 1/4 ∧ x4 = 1/4 :=
by sorry

end max_sum_product_sqrt_max_value_quarter_equality_condition_l3528_352824


namespace purchases_per_customer_l3528_352805

/-- Given a parking lot scenario, prove that each customer makes exactly one purchase. -/
theorem purchases_per_customer (num_cars : ℕ) (customers_per_car : ℕ) 
  (sports_sales : ℕ) (music_sales : ℕ) 
  (h1 : num_cars = 10) 
  (h2 : customers_per_car = 5) 
  (h3 : sports_sales = 20) 
  (h4 : music_sales = 30) : 
  (sports_sales + music_sales) / (num_cars * customers_per_car) = 1 :=
by sorry

end purchases_per_customer_l3528_352805


namespace least_seven_digit_binary_l3528_352834

/-- The least positive base ten number that requires seven digits for its binary representation -/
def leastSevenDigitBinary : ℕ := 64

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ := sorry

theorem least_seven_digit_binary :
  (∀ m : ℕ, m < leastSevenDigitBinary → binaryDigits m < 7) ∧
  binaryDigits leastSevenDigitBinary = 7 := by sorry

end least_seven_digit_binary_l3528_352834


namespace total_employees_l3528_352870

/-- Given a corporation with part-time and full-time employees, 
    calculate the total number of employees. -/
theorem total_employees (part_time full_time : ℕ) :
  part_time = 2041 →
  full_time = 63093 →
  part_time + full_time = 65134 := by
  sorry

end total_employees_l3528_352870


namespace floor_equation_solution_l3528_352856

theorem floor_equation_solution :
  ∀ m n : ℕ+,
  (⌊(m^2 : ℚ) / n⌋ + ⌊(n^2 : ℚ) / m⌋ = ⌊(m : ℚ) / n + (n : ℚ) / m⌋ + m * n) ↔ (m = 2 ∧ n = 1) :=
by sorry

end floor_equation_solution_l3528_352856


namespace perpendicular_line_through_point_l3528_352854

/-- Given a line l: 4x + 5y - 8 = 0 and a point A (3, 2), 
    the perpendicular line through A has the equation 4y - 5x + 7 = 0 -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let l : Set (ℝ × ℝ) := {(x, y) | 4 * x + 5 * y - 8 = 0}
  let A : ℝ × ℝ := (3, 2)
  let perpendicular_line : Set (ℝ × ℝ) := {(x, y) | 4 * y - 5 * x + 7 = 0}
  (∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ≠ q →
    (A.1 - p.1) * (q.1 - p.1) + (A.2 - p.2) * (q.2 - p.2) = 0) ∧
  A ∈ perpendicular_line :=
by sorry

end perpendicular_line_through_point_l3528_352854


namespace third_bakery_needs_twelve_sacks_l3528_352892

/-- The number of weeks Antoine supplies strawberries -/
def weeks : ℕ := 4

/-- The total number of sacks Antoine supplies in 4 weeks -/
def total_sacks : ℕ := 72

/-- The number of sacks the first bakery needs per week -/
def first_bakery_sacks : ℕ := 2

/-- The number of sacks the second bakery needs per week -/
def second_bakery_sacks : ℕ := 4

/-- The number of sacks the third bakery needs per week -/
def third_bakery_sacks : ℕ := total_sacks / weeks - (first_bakery_sacks + second_bakery_sacks)

theorem third_bakery_needs_twelve_sacks : third_bakery_sacks = 12 := by
  sorry

end third_bakery_needs_twelve_sacks_l3528_352892


namespace dice_roll_probability_l3528_352883

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of favorable outcomes on the first die (less than 3) -/
def favorableFirst : ℕ := 2

/-- The number of favorable outcomes on the second die (greater than 3) -/
def favorableSecond : ℕ := 3

/-- The probability of the desired outcome when rolling two dice -/
def probability : ℚ := (favorableFirst / numSides) * (favorableSecond / numSides)

theorem dice_roll_probability :
  probability = 1 / 6 := by
  sorry

end dice_roll_probability_l3528_352883


namespace effective_annual_rate_l3528_352836

/-- The effective annual compound interest rate for a 4-year investment -/
theorem effective_annual_rate (initial_investment final_amount : ℝ)
  (rate1 rate2 rate3 rate4 : ℝ) (h_initial : initial_investment = 810)
  (h_final : final_amount = 1550) (h_rate1 : rate1 = 0.05)
  (h_rate2 : rate2 = 0.07) (h_rate3 : rate3 = 0.06) (h_rate4 : rate4 = 0.04) :
  ∃ (r : ℝ), (abs (r - 0.1755) < 0.0001 ∧
  final_amount = initial_investment * ((1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4)) ∧
  final_amount = initial_investment * (1 + r)^4) :=
sorry

end effective_annual_rate_l3528_352836


namespace popsicles_left_l3528_352808

def initial_grape : ℕ := 2
def initial_cherry : ℕ := 13
def initial_banana : ℕ := 2
def initial_mango : ℕ := 8
def initial_strawberry : ℕ := 4
def initial_orange : ℕ := 6

def cherry_eaten : ℕ := 3
def grape_eaten : ℕ := 1

def total_initial : ℕ := initial_grape + initial_cherry + initial_banana + initial_mango + initial_strawberry + initial_orange

def total_eaten : ℕ := cherry_eaten + grape_eaten

theorem popsicles_left : total_initial - total_eaten = 31 := by
  sorry

end popsicles_left_l3528_352808


namespace students_in_sunghoons_class_l3528_352816

theorem students_in_sunghoons_class 
  (jisoo_students : ℕ) 
  (product : ℕ) 
  (h1 : jisoo_students = 36)
  (h2 : jisoo_students * sunghoon_students = product)
  (h3 : product = 1008) : 
  sunghoon_students = 28 :=
by
  sorry

end students_in_sunghoons_class_l3528_352816


namespace triangle_side_length_l3528_352833

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.α > 0 ∧ t.β > 0 ∧ t.γ > 0 ∧
  t.α + t.β + t.γ = Real.pi ∧
  3 * t.α + 2 * t.β = Real.pi

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h : TriangleProperties t) 
  (ha : t.a = 2) 
  (hb : t.b = 3) : 
  t.c = 4 := by
  sorry

end triangle_side_length_l3528_352833


namespace product_in_first_quadrant_l3528_352855

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

theorem product_in_first_quadrant :
  let z : ℂ := complex_multiply 1 3 3 (-1)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end product_in_first_quadrant_l3528_352855


namespace exists_points_with_midpoint_l3528_352881

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the midpoint of two points
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

-- Theorem statement
theorem exists_points_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
by
  sorry

end exists_points_with_midpoint_l3528_352881


namespace function_properties_l3528_352823

-- Define the function f
def f (a c : ℕ) (x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- State the theorem
theorem function_properties (a c : ℕ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (f a c 1 = 5) →
  (6 < f a c 2 ∧ f a c 2 < 11) →
  (a = 1 ∧ c = 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, f a c x - 2 * m * x ≤ 1) → m ≥ 1) :=
by sorry

end function_properties_l3528_352823


namespace complex_equality_squared_l3528_352877

theorem complex_equality_squared (m n : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : m * (1 + i) = 1 + n * i) : 
  ((m + n * i) / (m - n * i))^2 = -1 := by
  sorry

end complex_equality_squared_l3528_352877


namespace min_power_cycles_mod1024_l3528_352838

/-- A power cycle is a set of nonnegative integer powers of an integer a. -/
def PowerCycle (a : ℤ) : Set ℤ :=
  {k : ℤ | ∃ n : ℕ, k = a ^ n}

/-- A set of power cycles covers all odd integers modulo 1024 if for any odd integer n,
    there exists a power cycle in the set and an integer k in that cycle
    such that n ≡ k (mod 1024). -/
def CoverAllOddMod1024 (S : Set (Set ℤ)) : Prop :=
  ∀ n : ℤ, Odd n → ∃ C ∈ S, ∃ k ∈ C, n ≡ k [ZMOD 1024]

/-- The theorem states that the minimum number of power cycles required
    to cover all odd integers modulo 1024 is 10. -/
theorem min_power_cycles_mod1024 :
  ∃ S : Set (Set ℤ),
    (∀ C ∈ S, ∃ a : ℤ, C = PowerCycle a) ∧
    CoverAllOddMod1024 S ∧
    S.ncard = 10 ∧
    ∀ T : Set (Set ℤ),
      (∀ C ∈ T, ∃ a : ℤ, C = PowerCycle a) →
      CoverAllOddMod1024 T →
      T.ncard ≥ 10 :=
by sorry

end min_power_cycles_mod1024_l3528_352838


namespace ngon_area_division_l3528_352825

/-- Represents a convex n-gon -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry -- Condition for convexity

/-- Represents a point on the boundary of the n-gon -/
structure BoundaryPoint (n : ℕ) (polygon : ConvexNGon n) where
  point : ℝ × ℝ
  on_boundary : sorry -- Condition for being on the boundary
  not_vertex : sorry -- Condition for not being a vertex

/-- Predicate to check if a line divides the polygon's area in half -/
def divides_area_in_half (n : ℕ) (polygon : ConvexNGon n) (a b : ℝ × ℝ) : Prop := sorry

/-- The number of sides on which the boundary points lie -/
def sides_with_points (n : ℕ) (polygon : ConvexNGon n) (points : Fin n → BoundaryPoint n polygon) : ℕ := sorry

theorem ngon_area_division (n : ℕ) (polygon : ConvexNGon n) 
  (points : Fin n → BoundaryPoint n polygon)
  (h_divide : ∀ i : Fin n, divides_area_in_half n polygon (polygon.vertices i) (points i).point) :
  (3 ≤ sides_with_points n polygon points) ∧ 
  (sides_with_points n polygon points ≤ if n % 2 = 0 then n - 1 else n) := sorry

end ngon_area_division_l3528_352825


namespace right_triangle_acute_angles_l3528_352809

theorem right_triangle_acute_angles (α β : Real) : 
  α = 30 → β = 90 → ∃ γ : Real, γ = 60 ∧ α + β + γ = 180 :=
by sorry

end right_triangle_acute_angles_l3528_352809


namespace profit_calculation_l3528_352869

/-- The number of pencils purchased -/
def total_pencils : ℕ := 2000

/-- The purchase price per pencil in dollars -/
def purchase_price : ℚ := 1/5

/-- The selling price per pencil in dollars -/
def selling_price : ℚ := 2/5

/-- The desired profit in dollars -/
def desired_profit : ℚ := 160

/-- The number of pencils that must be sold to achieve the desired profit -/
def pencils_to_sell : ℕ := 1400

theorem profit_calculation :
  (pencils_to_sell : ℚ) * selling_price - (total_pencils : ℚ) * purchase_price = desired_profit :=
sorry

end profit_calculation_l3528_352869


namespace gcd_of_256_180_720_l3528_352832

theorem gcd_of_256_180_720 : Nat.gcd 256 (Nat.gcd 180 720) = 4 := by
  sorry

end gcd_of_256_180_720_l3528_352832


namespace cost_price_is_1000_l3528_352896

/-- The cost price of a toy, given the selling conditions -/
def cost_price_of_toy (total_sold : ℕ) (selling_price : ℕ) (gain_in_toys : ℕ) : ℕ :=
  selling_price / (total_sold + gain_in_toys)

/-- Theorem stating the cost price of a toy under the given conditions -/
theorem cost_price_is_1000 :
  cost_price_of_toy 18 21000 3 = 1000 := by
  sorry

end cost_price_is_1000_l3528_352896


namespace fraction_simplification_l3528_352804

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end fraction_simplification_l3528_352804


namespace polynomial_division_degree_l3528_352876

open Polynomial

theorem polynomial_division_degree (f d q r : ℝ[X]) : 
  degree f = 15 →
  f = d * q + r →
  degree q = 9 →
  degree r = 4 →
  degree r < degree d →
  degree d = 6 := by
sorry

end polynomial_division_degree_l3528_352876


namespace ceiling_sqrt_225_l3528_352813

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end ceiling_sqrt_225_l3528_352813


namespace power_product_equals_1938400_l3528_352886

theorem power_product_equals_1938400 : 2^4 * 3^2 * 5^2 * 7^2 * 11 = 1938400 := by
  sorry

end power_product_equals_1938400_l3528_352886


namespace train_length_l3528_352817

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 20 seconds,
    and the platform length is 285 meters, the length of the train is 300 meters. -/
theorem train_length (crossing_time_platform : ℝ) (crossing_time_pole : ℝ) (platform_length : ℝ)
    (h1 : crossing_time_platform = 39)
    (h2 : crossing_time_pole = 20)
    (h3 : platform_length = 285) :
    ∃ train_length : ℝ, train_length = 300 := by
  sorry

end train_length_l3528_352817


namespace min_distance_line_parabola_l3528_352801

/-- The minimum distance between a point on the line y = (15/8)x - 4 and a point on the parabola y = x^2 is 47/32 -/
theorem min_distance_line_parabola :
  let line := fun (x : ℝ) => (15/8) * x - 4
  let parabola := fun (x : ℝ) => x^2
  ∃ (x₁ x₂ : ℝ),
    (∀ (y₁ y₂ : ℝ),
      (line y₁ = (15/8) * y₁ - 4) →
      (parabola y₂ = y₂^2) →
      ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)^(1/2) ≤ ((y₂ - y₁)^2 + (parabola y₂ - line y₁)^2)^(1/2)) ∧
    ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)^(1/2) = 47/32 :=
by sorry

end min_distance_line_parabola_l3528_352801


namespace triangle_max_area_l3528_352873

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (Real.sin A - Real.sin B) / Real.sin C = (c - b) / (2 + b) →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧
    area = (1/2) * a * b * Real.sin C ∧
    ∀ (area' : ℝ), area' = (1/2) * a * b * Real.sin C → area' ≤ area :=
by sorry

end triangle_max_area_l3528_352873


namespace distance_A_O_min_distance_O_line_l3528_352849

-- Define the polyline distance function
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define point A
def A : ℝ × ℝ := (-1, 3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def on_line (x y : ℝ) : Prop :=
  2 * x + y - 2 * Real.sqrt 5 = 0

-- Theorem 1: The polyline distance between A and O is 4
theorem distance_A_O :
  polyline_distance A.1 A.2 O.1 O.2 = 4 := by sorry

-- Theorem 2: The minimum polyline distance between O and any point on the line is √5
theorem min_distance_O_line :
  ∃ (x y : ℝ), on_line x y ∧
  ∀ (x' y' : ℝ), on_line x' y' →
  polyline_distance O.1 O.2 x y ≤ polyline_distance O.1 O.2 x' y' ∧
  polyline_distance O.1 O.2 x y = Real.sqrt 5 := by sorry

end distance_A_O_min_distance_O_line_l3528_352849


namespace christen_peeled_21_potatoes_l3528_352802

/-- Calculates the number of potatoes Christen peeled --/
def christenPotatoesCount (totalPotatoes : ℕ) (homerRate : ℕ) (christenRate : ℕ) (timeBeforeJoin : ℕ) : ℕ :=
  let homerInitialPotatoes := homerRate * timeBeforeJoin
  let remainingPotatoes := totalPotatoes - homerInitialPotatoes
  let combinedRate := homerRate + christenRate
  let timeAfterJoin := remainingPotatoes / combinedRate
  christenRate * timeAfterJoin

theorem christen_peeled_21_potatoes :
  christenPotatoesCount 60 4 6 6 = 21 := by
  sorry

end christen_peeled_21_potatoes_l3528_352802


namespace total_books_count_l3528_352899

/-- The number of bookshelves -/
def num_bookshelves : ℕ := 1250

/-- The number of books on each bookshelf -/
def books_per_shelf : ℕ := 45

/-- The total number of books on all shelves -/
def total_books : ℕ := num_bookshelves * books_per_shelf

theorem total_books_count : total_books = 56250 := by
  sorry

end total_books_count_l3528_352899


namespace equation_solution_l3528_352885

theorem equation_solution :
  ∃ (a b c d : ℚ),
    a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0 ∧
    a = 1/5 ∧ b = 2/5 ∧ c = 3/5 ∧ d = 4/5 := by
  sorry

end equation_solution_l3528_352885


namespace trigonometric_product_sqrt_l3528_352874

theorem trigonometric_product_sqrt (h1 : Real.sin (π / 6) = 1 / 2)
                                   (h2 : Real.sin (π / 4) = Real.sqrt 2 / 2)
                                   (h3 : Real.sin (π / 3) = Real.sqrt 3 / 2) :
  Real.sqrt ((2 - (Real.sin (π / 6))^2) * (2 - (Real.sin (π / 4))^2) * (2 - (Real.sin (π / 3))^2)) = Real.sqrt 210 / 8 := by
  sorry

end trigonometric_product_sqrt_l3528_352874


namespace teddy_hamburger_count_l3528_352840

/-- The number of hamburgers Teddy bought -/
def teddy_hamburgers : ℕ := 5

/-- The total amount spent by Robert and Teddy -/
def total_spent : ℕ := 106

/-- The cost of a pizza box -/
def pizza_cost : ℕ := 10

/-- The cost of a soft drink -/
def drink_cost : ℕ := 2

/-- The cost of a hamburger -/
def hamburger_cost : ℕ := 3

/-- The number of pizza boxes Robert bought -/
def robert_pizza : ℕ := 5

/-- The number of soft drinks Robert bought -/
def robert_drinks : ℕ := 10

/-- The number of soft drinks Teddy bought -/
def teddy_drinks : ℕ := 10

theorem teddy_hamburger_count :
  total_spent = 
    robert_pizza * pizza_cost + 
    (robert_drinks + teddy_drinks) * drink_cost + 
    teddy_hamburgers * hamburger_cost := by
  sorry

end teddy_hamburger_count_l3528_352840


namespace segment_ratio_l3528_352868

/-- Given four distinct points on a plane with segments of lengths a, a, b, a+√3b, 2a, and 2b
    connecting them, the ratio of b to a is 2 + √3. -/
theorem segment_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    ({dist p1 p2, dist p1 p3, dist p1 p4, dist p2 p3, dist p2 p4, dist p3 p4} : Finset ℝ) = 
      {a, a, b, a + Real.sqrt 3 * b, 2 * a, 2 * b} →
    b / a = 2 + Real.sqrt 3 := by
  sorry

#check segment_ratio

end segment_ratio_l3528_352868


namespace parallelogram_area_18_16_l3528_352864

/-- The area of a parallelogram given its base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 16 cm is 288 square centimeters -/
theorem parallelogram_area_18_16 : 
  parallelogram_area 18 16 = 288 := by sorry

end parallelogram_area_18_16_l3528_352864


namespace michaels_dogs_l3528_352859

theorem michaels_dogs (num_cats : ℕ) (cost_per_animal : ℕ) (total_cost : ℕ) :
  num_cats = 2 →
  cost_per_animal = 13 →
  total_cost = 65 →
  ∃ num_dogs : ℕ, num_dogs = 3 ∧ total_cost = cost_per_animal * (num_cats + num_dogs) :=
by sorry

end michaels_dogs_l3528_352859


namespace smallest_prime_factor_of_2023_l3528_352891

theorem smallest_prime_factor_of_2023 : Nat.minFac 2023 = 7 := by
  sorry

end smallest_prime_factor_of_2023_l3528_352891


namespace sufficient_condition_for_inequality_l3528_352812

theorem sufficient_condition_for_inequality (a : ℝ) :
  0 < a ∧ a < (1/5) → (1/a) > 3 := by
  sorry

end sufficient_condition_for_inequality_l3528_352812


namespace total_vehicles_l3528_352884

theorem total_vehicles (lanes : Nat) (trucks_per_lane : Nat) : 
  lanes = 4 → 
  trucks_per_lane = 60 → 
  (lanes * trucks_per_lane * 2 + lanes * trucks_per_lane) = 2160 :=
by
  sorry

end total_vehicles_l3528_352884


namespace four_digit_number_problem_l3528_352844

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_mean (x y z : ℕ) : Prop :=
  x * z = y * y

theorem four_digit_number_problem (a b c d : ℕ) :
  a ≥ 1 ∧ a ≤ 9 ∧
  b ≥ 0 ∧ b ≤ 9 ∧
  c ≥ 0 ∧ c ≤ 9 ∧
  d ≥ 0 ∧ d ≤ 9 ∧
  is_arithmetic_sequence a b c d ∧
  is_geometric_mean a b d ∧
  1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a = 11110 →
  (a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5) ∨ (a = 2 ∧ b = 4 ∧ c = 6 ∧ d = 8) :=
by sorry

end four_digit_number_problem_l3528_352844


namespace quotient_negative_one_sum_zero_l3528_352863

theorem quotient_negative_one_sum_zero (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / b = -1 → a + b = 0 := by
  sorry

end quotient_negative_one_sum_zero_l3528_352863


namespace intersection_equals_B_B_proper_superset_A_l3528_352898

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

-- Theorem for part 1
theorem intersection_equals_B (m : ℝ) : (A ∩ B m) = B m ↔ m ≤ 1 := by sorry

-- Theorem for part 2
theorem B_proper_superset_A (m : ℝ) : A ⊂ B m ↔ m ≥ 2 := by sorry

end intersection_equals_B_B_proper_superset_A_l3528_352898


namespace triple_base_double_exponent_l3528_352862

theorem triple_base_double_exponent (a b x : ℝ) (hb : b ≠ 0) :
  let r := (3 * a) ^ (2 * b)
  r = a ^ b * x ^ b → x = 9 * a := by
sorry

end triple_base_double_exponent_l3528_352862


namespace gus_ate_fourteen_eggs_l3528_352806

/-- Represents the number of eggs in each dish Gus ate throughout the day -/
def eggs_per_dish : List Nat := [2, 1, 3, 2, 1, 2, 3]

/-- The total number of eggs Gus ate -/
def total_eggs : Nat := eggs_per_dish.sum

/-- Theorem stating that the total number of eggs Gus ate is 14 -/
theorem gus_ate_fourteen_eggs : total_eggs = 14 := by sorry

end gus_ate_fourteen_eggs_l3528_352806


namespace garden_area_increase_l3528_352842

/-- Proves that changing a rectangular garden to a square garden with the same perimeter increases the area by 100 square feet. -/
theorem garden_area_increase (length width : ℝ) (h1 : length = 40) (h2 : width = 20) :
  let rectangle_area := length * width
  let perimeter := 2 * (length + width)
  let square_side := perimeter / 4
  let square_area := square_side * square_side
  square_area - rectangle_area = 100 := by
  sorry

end garden_area_increase_l3528_352842


namespace problem_solution_l3528_352875

def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

theorem problem_solution :
  (∀ x, x ∈ (Set.univ \ M) ∪ (N 2) ↔ x > 5 ∨ x < -2) ∧
  (∀ a, M ∪ N a = M ↔ a < -2 ∨ (-1 ≤ a ∧ a ≤ 2)) :=
sorry

end problem_solution_l3528_352875


namespace sin_330_degrees_l3528_352821

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by sorry

end sin_330_degrees_l3528_352821


namespace existence_of_solution_l3528_352831

theorem existence_of_solution : ∃ (a b : ℕ), 
  a > 1 ∧ b > 1 ∧ a^13 * b^31 = 6^2015 ∧ a = 2^155 ∧ b = 3^65 := by
  sorry

end existence_of_solution_l3528_352831


namespace arithmetic_sequence_sum_l3528_352830

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 = 25 →
  b 1 = 75 →
  a 2 + b 2 = 100 →
  a 37 + b 37 = 100 := by
  sorry

end arithmetic_sequence_sum_l3528_352830


namespace system_solution_l3528_352853

theorem system_solution (x y z : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^3 = 2*y^2 - z ∧
  y^3 = 2*z^2 - x ∧
  z^3 = 2*x^2 - y →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end system_solution_l3528_352853


namespace typists_for_180_letters_l3528_352811

/-- The number of typists needed to type a certain number of letters in a given time, 
    given a known typing rate. -/
def typists_needed 
  (known_typists : ℕ) 
  (known_letters : ℕ) 
  (known_minutes : ℕ) 
  (target_letters : ℕ) 
  (target_minutes : ℕ) : ℕ :=
  sorry

theorem typists_for_180_letters 
  (h1 : typists_needed 20 40 20 180 60 = 30) : 
  typists_needed 20 40 20 180 60 = 30 := by
  sorry

end typists_for_180_letters_l3528_352811


namespace conic_section_is_hyperbola_l3528_352800

/-- The conic section represented by the equation (2x-7)^2 - 4(y+3)^2 = 169 is a hyperbola. -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (2*x - 7)^2 - 4*(y + 3)^2 = 169 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0) ∧
    (a > 0 ∧ b < 0) := by
  sorry

end conic_section_is_hyperbola_l3528_352800


namespace maxwell_brad_meeting_time_l3528_352803

theorem maxwell_brad_meeting_time 
  (distance : ℝ) 
  (maxwell_speed : ℝ) 
  (brad_speed : ℝ) 
  (maxwell_head_start : ℝ) :
  distance = 34 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_head_start = 1 →
  ∃ (total_time : ℝ), 
    total_time = 4 ∧
    maxwell_speed * total_time + brad_speed * (total_time - maxwell_head_start) = distance :=
by
  sorry

end maxwell_brad_meeting_time_l3528_352803


namespace tenth_term_of_arithmetic_sequence_l3528_352861

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 5) :
  a 10 = 19 := by
sorry

end tenth_term_of_arithmetic_sequence_l3528_352861


namespace saltwater_aquariums_l3528_352872

theorem saltwater_aquariums (total_saltwater_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_saltwater_animals = 1012)
  (h2 : animals_per_aquarium = 46) :
  total_saltwater_animals / animals_per_aquarium = 22 := by
  sorry

end saltwater_aquariums_l3528_352872


namespace ratio_problem_l3528_352860

theorem ratio_problem (first_term second_term : ℚ) : 
  first_term = 15 → 
  first_term / second_term = 60 / 100 → 
  second_term = 25 := by
sorry

end ratio_problem_l3528_352860
