import Mathlib

namespace NUMINAMATH_CALUDE_kennel_dogs_l376_37672

/-- Given a kennel with cats and dogs, prove the number of dogs. -/
theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 2 / 3 →  -- ratio of cats to dogs is 2:3
  cats = dogs - 6 →            -- 6 fewer cats than dogs
  dogs = 18 := by              -- prove that there are 18 dogs
sorry

end NUMINAMATH_CALUDE_kennel_dogs_l376_37672


namespace NUMINAMATH_CALUDE_als_original_portion_l376_37602

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3*b + 3*c = 1800 →
  a = 825 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l376_37602


namespace NUMINAMATH_CALUDE_son_age_proof_l376_37633

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l376_37633


namespace NUMINAMATH_CALUDE_divisibility_condition_l376_37627

theorem divisibility_condition (n : ℤ) : 
  (n^5 + 3) % (n^2 + 1) = 0 ↔ n ∈ ({-3, -1, 0, 1, 2} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l376_37627


namespace NUMINAMATH_CALUDE_olivia_car_rental_cost_l376_37630

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Proves that Olivia's car rental costs $215 given the specified conditions. -/
theorem olivia_car_rental_cost :
  carRentalCost 30 (1/4) 3 500 = 215 := by
  sorry

end NUMINAMATH_CALUDE_olivia_car_rental_cost_l376_37630


namespace NUMINAMATH_CALUDE_complex_power_sum_l376_37650

theorem complex_power_sum (z : ℂ) (h : z = -Complex.I) : z^100 + z^50 + 1 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l376_37650


namespace NUMINAMATH_CALUDE_lottery_expected_wins_l376_37640

/-- A lottery with a winning probability of 1/4 -/
structure Lottery where
  win_prob : ℝ
  win_prob_eq : win_prob = 1/4

/-- The expected number of winning tickets when drawing n tickets -/
def expected_wins (L : Lottery) (n : ℕ) : ℝ := n * L.win_prob

theorem lottery_expected_wins (L : Lottery) : expected_wins L 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lottery_expected_wins_l376_37640


namespace NUMINAMATH_CALUDE_jellybean_count_l376_37608

def jellybean_problem (initial : ℕ) (first_removal : ℕ) (added_back : ℕ) (second_removal : ℕ) : ℕ :=
  initial - first_removal + added_back - second_removal

theorem jellybean_count : jellybean_problem 37 15 5 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l376_37608


namespace NUMINAMATH_CALUDE_initial_fish_l376_37632

def fish_bought : ℝ := 280.0
def fish_now : ℕ := 492

theorem initial_fish : ℕ := by
  sorry

#check initial_fish = 212

end NUMINAMATH_CALUDE_initial_fish_l376_37632


namespace NUMINAMATH_CALUDE_expression_evaluation_l376_37680

theorem expression_evaluation (x : ℝ) (h : x = 1.25) :
  (3 * x^2 - 8 * x + 2) * (4 * x - 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l376_37680


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l376_37677

theorem fraction_equation_solution : 
  ∀ (A B : ℚ), 
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 5 → 
    (B * x - 13) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) → 
  A + B = 31/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l376_37677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l376_37664

/-- An arithmetic sequence with given start, end, and common difference -/
def arithmetic_sequence (start end_ diff : ℕ) : List ℕ :=
  let n := (end_ - start) / diff + 1
  List.range n |>.map (fun i => start + i * diff)

/-- The problem statement -/
theorem arithmetic_sequence_length :
  (arithmetic_sequence 20 150 5).length = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l376_37664


namespace NUMINAMATH_CALUDE_xy_inequality_l376_37693

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 - x*y = 1) : 
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l376_37693


namespace NUMINAMATH_CALUDE_closest_to_10_l376_37691

def numbers : List ℝ := [9.998, 10.1, 10.09, 10.001]

def distance_to_10 (x : ℝ) : ℝ := |x - 10|

theorem closest_to_10 : 
  ∀ x ∈ numbers, distance_to_10 10.001 ≤ distance_to_10 x :=
by sorry

end NUMINAMATH_CALUDE_closest_to_10_l376_37691


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l376_37649

/-- Parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {(x, y) | y = x^2}

/-- Point Q -/
def Q : ℝ × ℝ := (10, 4)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) := {(x, y) | y - 4 = m * (x - 10)}

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop := line_through_Q m ∩ P = ∅

theorem parabola_line_intersection (r s : ℝ) :
  (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l376_37649


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l376_37655

theorem complex_expression_simplification :
  let i : ℂ := Complex.I
  7 * (4 - 2*i) + 4*i*(7 - 3*i) + 2*(5 + i) = 50 + 16*i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l376_37655


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l376_37626

/-- Given a boat that travels 32 km along a stream and 12 km against the same stream
    in one hour each, its speed in still water is 22 km/hr. -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 32) 
  (h2 : against_stream = 12) : 
  (along_stream + against_stream) / 2 = 22 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l376_37626


namespace NUMINAMATH_CALUDE_sally_balloons_l376_37669

/-- The number of blue balloons each person has -/
structure Balloons where
  joan : ℕ
  sally : ℕ
  jessica : ℕ

/-- The total number of blue balloons -/
def total_balloons (b : Balloons) : ℕ := b.joan + b.sally + b.jessica

/-- Theorem stating Sally's number of balloons -/
theorem sally_balloons (b : Balloons) 
  (h1 : b.joan = 9)
  (h2 : b.jessica = 2)
  (h3 : total_balloons b = 16) :
  b.sally = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_balloons_l376_37669


namespace NUMINAMATH_CALUDE_strawberries_in_buckets_l376_37643

theorem strawberries_in_buckets
  (total_strawberries : ℕ)
  (num_buckets : ℕ)
  (removed_per_bucket : ℕ)
  (h1 : total_strawberries = 300)
  (h2 : num_buckets = 5)
  (h3 : removed_per_bucket = 20)
  : (total_strawberries / num_buckets) - removed_per_bucket = 40 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_in_buckets_l376_37643


namespace NUMINAMATH_CALUDE_water_level_unchanged_l376_37629

-- Define the densities of water and ice
variable (ρ_water ρ_ice : ℝ)

-- Define the initial volume of water taken for freezing
variable (V : ℝ)

-- Hypothesis: density of ice is less than density of water
axiom h1 : ρ_ice < ρ_water

-- Hypothesis: mass is conserved when water freezes
axiom h2 : V * ρ_water = (V * ρ_water / ρ_ice) * ρ_ice

-- Hypothesis: Archimedes' principle applies to floating ice
axiom h3 : ∀ W : ℝ, W * ρ_ice = (W * ρ_ice / ρ_water) * ρ_water

-- Theorem: The volume of water displaced by the ice is equal to the original volume of water
theorem water_level_unchanged (V : ℝ) (h1 : ρ_ice < ρ_water) 
  (h2 : V * ρ_water = (V * ρ_water / ρ_ice) * ρ_ice) 
  (h3 : ∀ W : ℝ, W * ρ_ice = (W * ρ_ice / ρ_water) * ρ_water) :
  (V * ρ_water / ρ_ice) * ρ_ice / ρ_water = V :=
by sorry

end NUMINAMATH_CALUDE_water_level_unchanged_l376_37629


namespace NUMINAMATH_CALUDE_expression_equals_two_power_thirty_l376_37681

theorem expression_equals_two_power_thirty :
  (((16^16 / 16^14)^3 * 8^6) / 2^12) = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_power_thirty_l376_37681


namespace NUMINAMATH_CALUDE_circle_center_first_quadrant_l376_37653

theorem circle_center_first_quadrant (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x + (2*m - 2)*y + 2*m^2 = 0 →
    ∃ r : ℝ, (x - m)^2 + (y - (1 - m))^2 = r^2) →
  (m > 0 ∧ 1 - m > 0) →
  0 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_first_quadrant_l376_37653


namespace NUMINAMATH_CALUDE_base6_addition_subtraction_l376_37697

/-- Converts a base 6 number to its decimal representation -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base 6 representation -/
def decimalToBase6 (n : ℕ) : ℕ := sorry

theorem base6_addition_subtraction :
  decimalToBase6 ((base6ToDecimal 35 + base6ToDecimal 14) - base6ToDecimal 20) = 33 := by sorry

end NUMINAMATH_CALUDE_base6_addition_subtraction_l376_37697


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l376_37652

theorem sufficient_condition_for_inequality (a : ℝ) :
  a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l376_37652


namespace NUMINAMATH_CALUDE_plane_equation_l376_37660

/-- A plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  coprime : Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

def parallel (p1 p2 : Plane) : Prop :=
  p1.a * p2.b = p1.b * p2.a ∧ p1.a * p2.c = p1.c * p2.a ∧ p1.b * p2.c = p1.c * p2.b

def passes_through (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem plane_equation : 
  ∃ (plane : Plane), 
    plane.a = 3 ∧ 
    plane.b = -4 ∧ 
    plane.c = 1 ∧ 
    plane.d = 7 ∧ 
    passes_through plane ⟨2, 3, -1⟩ ∧ 
    parallel plane ⟨3, -4, 1, -5, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l376_37660


namespace NUMINAMATH_CALUDE_correct_num_technicians_l376_37604

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technicians -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians : 
  num_technicians * avg_salary_technicians + 
  (total_workers - num_technicians) * avg_salary_rest = 
  total_workers * avg_salary_all :=
sorry

#check correct_num_technicians

end NUMINAMATH_CALUDE_correct_num_technicians_l376_37604


namespace NUMINAMATH_CALUDE_max_value_of_function_l376_37636

theorem max_value_of_function (x : ℝ) : 
  (∀ x, -1 ≤ Real.cos x ∧ Real.cos x ≤ 1) → 
  ∃ y_max : ℝ, y_max = 4 ∧ ∀ x, 3 - Real.cos (x / 2) ≤ y_max := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l376_37636


namespace NUMINAMATH_CALUDE_relationship_between_A_B_C_l376_37663

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the relationships between A, B, and C
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary_and_sufficient (P Q : Prop) : Prop :=
  (P ↔ Q)

-- Theorem statement
theorem relationship_between_A_B_C
  (h1 : sufficient_not_necessary A B)
  (h2 : necessary_and_sufficient B C) :
  sufficient_not_necessary C A :=
sorry

end NUMINAMATH_CALUDE_relationship_between_A_B_C_l376_37663


namespace NUMINAMATH_CALUDE_sin_B_range_in_acute_triangle_l376_37665

theorem sin_B_range_in_acute_triangle (A B C : Real) (a b c : Real) (S : Real) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  S = (1 / 2) * b * c * Real.sin A →
  a^2 = 2 * S + (b - c)^2 →
  3 / 5 < Real.sin B ∧ Real.sin B < 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_B_range_in_acute_triangle_l376_37665


namespace NUMINAMATH_CALUDE_third_median_length_special_triangle_third_median_l376_37616

/-- A triangle with specific median lengths and area -/
structure SpecialTriangle where
  -- Two medians of the triangle
  median1 : ℝ
  median2 : ℝ
  -- Area of the triangle
  area : ℝ
  -- Conditions on the medians and area
  median1_length : median1 = 4
  median2_length : median2 = 8
  triangle_area : area = 4 * Real.sqrt 15

/-- The third median of the special triangle has length 7 -/
theorem third_median_length (t : SpecialTriangle) : ℝ :=
  7

/-- The theorem stating that the third median of the special triangle has length 7 -/
theorem special_triangle_third_median (t : SpecialTriangle) : 
  third_median_length t = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_median_length_special_triangle_third_median_l376_37616


namespace NUMINAMATH_CALUDE_last_two_digits_product_l376_37641

theorem last_two_digits_product (A B : ℕ) : 
  (A * 10 + B) % 6 = 0 → A + B = 11 → A * B = 24 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l376_37641


namespace NUMINAMATH_CALUDE_two_integers_sum_l376_37659

theorem two_integers_sum (x y : ℕ+) : x - y = 4 → x * y = 63 → x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l376_37659


namespace NUMINAMATH_CALUDE_paint_per_statue_l376_37615

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 3/6)
  (h2 : num_statues = 3) :
  total_paint / num_statues = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l376_37615


namespace NUMINAMATH_CALUDE_zayne_revenue_l376_37634

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  single_price : ℕ  -- Price of a single bracelet
  pair_price : ℕ    -- Price of a pair of bracelets
  initial_stock : ℕ -- Initial number of bracelets
  single_sale_revenue : ℕ -- Revenue from selling single bracelets

/-- Calculates the total revenue from selling bracelets -/
def total_revenue (sale : BraceletSale) : ℕ :=
  let single_bracelets_sold := sale.single_sale_revenue / sale.single_price
  let remaining_bracelets := sale.initial_stock - single_bracelets_sold
  let pairs_sold := remaining_bracelets / 2
  let pair_revenue := pairs_sold * sale.pair_price
  sale.single_sale_revenue + pair_revenue

/-- Theorem stating that Zayne's total revenue is $132 -/
theorem zayne_revenue :
  ∃ (sale : BraceletSale),
    sale.single_price = 5 ∧
    sale.pair_price = 8 ∧
    sale.initial_stock = 30 ∧
    sale.single_sale_revenue = 60 ∧
    total_revenue sale = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_zayne_revenue_l376_37634


namespace NUMINAMATH_CALUDE_weekdays_wearing_one_shirt_to_school_l376_37646

def shirts_for_two_weeks : ℕ := 22

def after_school_club_days_per_week : ℕ := 3
def saturdays_per_week : ℕ := 1
def sundays_per_week : ℕ := 1
def weeks : ℕ := 2

def shirts_for_after_school_club : ℕ := after_school_club_days_per_week * weeks
def shirts_for_saturdays : ℕ := saturdays_per_week * weeks
def shirts_for_sundays : ℕ := 2 * sundays_per_week * weeks

def shirts_for_other_activities : ℕ := 
  shirts_for_after_school_club + shirts_for_saturdays + shirts_for_sundays

theorem weekdays_wearing_one_shirt_to_school : 
  (shirts_for_two_weeks - shirts_for_other_activities) / weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_weekdays_wearing_one_shirt_to_school_l376_37646


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l376_37654

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l376_37654


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l376_37698

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≥ 0) :
  x^(3/2) + 6*x^(5/4) + 8*x^(3/4) ≥ 15*x ∧
  (x^(3/2) + 6*x^(5/4) + 8*x^(3/4) = 15*x ↔ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l376_37698


namespace NUMINAMATH_CALUDE_no_solution_exists_l376_37668

theorem no_solution_exists : ¬∃ x : ℝ, 1000^2 + 1001^2 + 1002^2 + x^2 + 1004^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l376_37668


namespace NUMINAMATH_CALUDE_new_girl_weight_l376_37644

theorem new_girl_weight (initial_total_weight : ℝ) : 
  let initial_average := initial_total_weight / 10
  let new_average := initial_average + 5
  let new_total_weight := new_average * 10
  new_total_weight = initial_total_weight - 50 + 100 := by sorry

end NUMINAMATH_CALUDE_new_girl_weight_l376_37644


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_expression_sum_l376_37651

theorem sum_of_roots_quadratic : ∀ (a b c : ℝ), a ≠ 0 →
  let roots := {x : ℝ | a * x^2 + b * x + c = 0}
  (∃ x₁ x₂, roots = {x₁, x₂}) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) → s = -b / a :=
by sorry

theorem undefined_expression_sum : 
  let roots := {x : ℝ | x^2 - 7*x + 12 = 0}
  (∃ x₁ x₂, roots = {x₁, x₂}) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) →
  s = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_undefined_expression_sum_l376_37651


namespace NUMINAMATH_CALUDE_vector_addition_l376_37675

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The sum of two vectors is equal to the vector from the start of the first to the end of the second. -/
theorem vector_addition (a b : V) :
  ∃ c : V, a + b = c ∧ ∃ (x y : V), x + a = y ∧ y + b = x + c :=
sorry

end NUMINAMATH_CALUDE_vector_addition_l376_37675


namespace NUMINAMATH_CALUDE_inequality_solution_set_l376_37689

theorem inequality_solution_set (x : ℝ) : 
  (abs (x - 1) + abs (x - 2) < 2) ↔ (1/2 < x ∧ x < 5/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l376_37689


namespace NUMINAMATH_CALUDE_aston_comic_pages_l376_37674

/-- The number of pages in each comic -/
def pages_per_comic : ℕ := 25

/-- The number of untorn comics initially in the box -/
def initial_comics : ℕ := 5

/-- The total number of comics in the box after Aston put them back together -/
def final_comics : ℕ := 11

/-- The number of pages Aston found on the floor -/
def pages_found : ℕ := (final_comics - initial_comics) * pages_per_comic

theorem aston_comic_pages : pages_found = 150 := by
  sorry

end NUMINAMATH_CALUDE_aston_comic_pages_l376_37674


namespace NUMINAMATH_CALUDE_no_integer_solution_l376_37600

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l376_37600


namespace NUMINAMATH_CALUDE_rice_distribution_theorem_l376_37631

/-- Represents the amount of rice in a container after dividing the total rice equally -/
def rice_per_container (total_pounds : ℚ) (num_containers : ℕ) : ℚ :=
  (total_pounds * 16) / num_containers

/-- Theorem stating that dividing 49 and 3/4 pounds of rice equally among 7 containers 
    results in approximately 114 ounces of rice per container -/
theorem rice_distribution_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |rice_per_container (49 + 3/4) 7 - 114| < ε :=
sorry

end NUMINAMATH_CALUDE_rice_distribution_theorem_l376_37631


namespace NUMINAMATH_CALUDE_mandy_toys_count_mandy_toys_count_proof_l376_37617

theorem mandy_toys_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mandy anna amanda peter =>
    anna = 3 * mandy ∧
    amanda = anna + 2 ∧
    peter = 2 * anna ∧
    mandy + anna + amanda + peter = 278 →
    mandy = 21

-- The proof is omitted
theorem mandy_toys_count_proof : mandy_toys_count 21 63 65 126 := by
  sorry

end NUMINAMATH_CALUDE_mandy_toys_count_mandy_toys_count_proof_l376_37617


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l376_37685

theorem arithmetic_mean_of_fractions : 
  let a := 9/12
  let b := 5/6
  let c := 7/8
  b = (a + c) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l376_37685


namespace NUMINAMATH_CALUDE_problem_1_l376_37696

theorem problem_1 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / y - y / x - (x^2 + y^2) / (x * y) = -2 * y / x :=
sorry

end NUMINAMATH_CALUDE_problem_1_l376_37696


namespace NUMINAMATH_CALUDE_log_43_between_consecutive_integers_l376_37684

theorem log_43_between_consecutive_integers : 
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 43 / Real.log 10 ∧ Real.log 43 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_43_between_consecutive_integers_l376_37684


namespace NUMINAMATH_CALUDE_container_volume_ratio_l376_37637

theorem container_volume_ratio :
  ∀ (A B C : ℝ),
  A > 0 → B > 0 → C > 0 →
  (3/4 * A - 5/8 * B = 7/8 * C - 1/2 * C) →
  (A / C = 4/5) :=
by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l376_37637


namespace NUMINAMATH_CALUDE_veranda_area_l376_37683

/-- Given a rectangular room with length 17 m and width 12 m, surrounded by a veranda of width 2 m on all sides, the area of the veranda is 132 square meters. -/
theorem veranda_area (room_length : ℝ) (room_width : ℝ) (veranda_width : ℝ) :
  room_length = 17 →
  room_width = 12 →
  veranda_width = 2 →
  let total_length := room_length + 2 * veranda_width
  let total_width := room_width + 2 * veranda_width
  let total_area := total_length * total_width
  let room_area := room_length * room_width
  total_area - room_area = 132 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l376_37683


namespace NUMINAMATH_CALUDE_special_parallelogram_segment_lengths_l376_37624

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ
  h_side1 : side1 = 5
  h_side2 : side2 = 13
  h_angle : angle = Real.arccos (6/13)

/-- The property of being divided into four equal quadrilaterals by perpendicular lines -/
def hasFourEqualQuadrilaterals (p : SpecialParallelogram) : Prop := sorry

/-- The lengths of segments created by the perpendicular lines -/
def segmentLengths (p : SpecialParallelogram) : ℝ × ℝ := sorry

/-- Theorem statement -/
theorem special_parallelogram_segment_lengths (p : SpecialParallelogram) 
  (h : hasFourEqualQuadrilaterals p) : 
  segmentLengths p = (3, 39/5) := by sorry


end NUMINAMATH_CALUDE_special_parallelogram_segment_lengths_l376_37624


namespace NUMINAMATH_CALUDE_amount_r_holds_l376_37661

theorem amount_r_holds (total : ℝ) (r_fraction : ℝ) (r_amount : ℝ) : 
  total = 7000 →
  r_fraction = 2/3 →
  r_amount = r_fraction * (total / (1 + r_fraction)) →
  r_amount = 2800 := by
sorry

end NUMINAMATH_CALUDE_amount_r_holds_l376_37661


namespace NUMINAMATH_CALUDE_tonys_fever_degree_l376_37605

/-- Proves that Tony's temperature is 5 degrees above the fever threshold given the conditions --/
theorem tonys_fever_degree (normal_temp : ℝ) (temp_increase : ℝ) (fever_threshold : ℝ) :
  normal_temp = 95 →
  temp_increase = 10 →
  fever_threshold = 100 →
  normal_temp + temp_increase - fever_threshold = 5 := by
  sorry

end NUMINAMATH_CALUDE_tonys_fever_degree_l376_37605


namespace NUMINAMATH_CALUDE_sin_135_degrees_l376_37686

theorem sin_135_degrees : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l376_37686


namespace NUMINAMATH_CALUDE_lucy_groceries_l376_37601

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 4

/-- The number of packs of cake Lucy bought -/
def cake : ℕ := 22

/-- The number of packs of chocolate Lucy bought -/
def chocolate : ℕ := 16

/-- The total number of packs of groceries Lucy bought -/
def total_groceries : ℕ := cookies + cake + chocolate

theorem lucy_groceries : total_groceries = 42 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l376_37601


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l376_37690

/-- The sum of the infinite series ∑(n=1 to ∞) 1/(n(n+3)) is equal to 11/18. -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' (n : ℕ), 1 / (n * (n + 3)) = 11 / 18 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l376_37690


namespace NUMINAMATH_CALUDE_square_sum_from_linear_and_product_l376_37666

theorem square_sum_from_linear_and_product (x y : ℝ) 
  (h1 : x + 3 * y = 3) (h2 : x * y = -6) : 
  x^2 + 9 * y^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_linear_and_product_l376_37666


namespace NUMINAMATH_CALUDE_unique_positive_solution_l376_37610

theorem unique_positive_solution :
  ∃! (y : ℝ), y > 0 ∧ (y - 6) / 12 = 6 / (y - 12) ∧ y = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l376_37610


namespace NUMINAMATH_CALUDE_mariel_dogs_count_l376_37638

theorem mariel_dogs_count (total_legs : ℕ) (other_dogs : ℕ) (human_legs : ℕ) (dog_legs : ℕ) :
  total_legs = 36 →
  other_dogs = 3 →
  human_legs = 2 →
  dog_legs = 4 →
  ∃ (mariel_dogs : ℕ), mariel_dogs = 5 ∧
    total_legs = 2 * human_legs + other_dogs * dog_legs + mariel_dogs * dog_legs :=
by
  sorry

end NUMINAMATH_CALUDE_mariel_dogs_count_l376_37638


namespace NUMINAMATH_CALUDE_complex_equation_solution_l376_37609

theorem complex_equation_solution (z : ℂ) (p q : ℝ) : 
  (∃ b : ℝ, z = Complex.I * b) →  -- z is purely imaginary
  (∃ c : ℝ, (z + 2)^2 + Complex.I * 8 = Complex.I * c) →  -- (z+2)^2 + 8i is purely imaginary
  2 * (z - 1)^2 + p * (z - 1) + q = 0 →  -- z-1 is a root of 2x^2 + px + q = 0
  z = Complex.I * 2 ∧ p = 4 ∧ q = 10 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l376_37609


namespace NUMINAMATH_CALUDE_candidates_per_state_l376_37667

theorem candidates_per_state (total : ℕ) (selected_A selected_B : ℕ) 
  (h1 : selected_A = total * 6 / 100)
  (h2 : selected_B = total * 7 / 100)
  (h3 : selected_B = selected_A + 79) :
  total = 7900 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l376_37667


namespace NUMINAMATH_CALUDE_arrangements_theorem_l376_37657

def num_men : ℕ := 5
def num_women : ℕ := 2
def positions_for_man_a : ℕ := 2

def arrangements_count : ℕ :=
  positions_for_man_a * Nat.factorial (num_men - 1 + 1) * Nat.factorial num_women

theorem arrangements_theorem : arrangements_count = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l376_37657


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l376_37670

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l376_37670


namespace NUMINAMATH_CALUDE_min_value_a_min_value_a_achievable_l376_37694

theorem min_value_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) → 
  a ≥ 4 :=
by sorry

theorem min_value_a_achievable : 
  ∃ a : ℝ, a = 4 ∧ (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_a_achievable_l376_37694


namespace NUMINAMATH_CALUDE_probability_product_216_l376_37687

/-- A standard die has 6 faces numbered from 1 to 6. -/
def StandardDie : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a standard die. -/
def DieProbability (event : Finset ℕ) : ℚ :=
  event.card / StandardDie.card

/-- The product of three numbers obtained from rolling three standard dice. -/
def ThreeDiceProduct (a b c : ℕ) : ℕ := a * b * c

/-- The event of rolling three sixes on three standard dice. -/
def ThreeSixes : Finset (ℕ × ℕ × ℕ) :=
  {(6, 6, 6)}

theorem probability_product_216 :
  DieProbability (ThreeSixes.image (fun (a, b, c) => ThreeDiceProduct a b c)) = 1 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_product_216_l376_37687


namespace NUMINAMATH_CALUDE_stable_table_configurations_l376_37614

def stableConfigurations (n : ℕ+) : ℕ :=
  (1/3) * (n+1) * (2*n^2 + 4*n + 3)

theorem stable_table_configurations (n : ℕ+) :
  (stableConfigurations n) =
  (Finset.sum (Finset.range (2*n+1)) (λ k =>
    (if k ≤ n then k + 1 else 2*n - k + 1)^2)) :=
sorry

end NUMINAMATH_CALUDE_stable_table_configurations_l376_37614


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l376_37647

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (m + 3) / (x - 1) = 1) → 
  (m > -4 ∧ m ≠ -3) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l376_37647


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l376_37695

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 = 2

/-- The focal length of an ellipse -/
def focal_length : ℝ := 4

/-- Theorem: The focal length of the ellipse defined by x^2/2 + y^2/4 = 2 is equal to 4 -/
theorem ellipse_focal_length :
  ∀ x y : ℝ, ellipse_equation x y → focal_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l376_37695


namespace NUMINAMATH_CALUDE_stock_value_change_l376_37628

/-- Theorem: Stock Value Change over Two Days
    Given a stock that decreases in value by 25% on the first day and
    increases by 40% on the second day, prove that the overall
    percentage change is a 5% increase. -/
theorem stock_value_change (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  (day2_value - initial_value) / initial_value = 0.05 := by
sorry

end NUMINAMATH_CALUDE_stock_value_change_l376_37628


namespace NUMINAMATH_CALUDE_mary_younger_than_albert_l376_37678

/-- Proves that Mary is 10 years younger than Albert given the conditions -/
theorem mary_younger_than_albert (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 5 →
  albert_age - mary_age = 10 := by
sorry

end NUMINAMATH_CALUDE_mary_younger_than_albert_l376_37678


namespace NUMINAMATH_CALUDE_inverse_proportion_l376_37688

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 30 and x - y = 10, then y = 50 when x = 4 -/
theorem inverse_proportion (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k)
    (h2 : x + y = 30) (h3 : x - y = 10) : 
    x = 4 → y = 50 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l376_37688


namespace NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l376_37671

/-- A polygon with interior angles summing to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_is_octagon :
  ∀ n : ℕ, (n - 2) * 180 = 1080 → n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l376_37671


namespace NUMINAMATH_CALUDE_sunzi_problem_l376_37645

theorem sunzi_problem (x y : ℚ) : 
  (x + (1/2) * y = 48 ∧ y + (2/3) * x = 48) ↔ 
  (x + (1/2) * y = 48 ∧ y + (2/3) * x = 48) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_problem_l376_37645


namespace NUMINAMATH_CALUDE_courtyard_length_l376_37642

theorem courtyard_length (width : ℝ) (tiles_per_sqft : ℝ) 
  (green_ratio : ℝ) (red_ratio : ℝ) (green_cost : ℝ) (red_cost : ℝ) 
  (total_cost : ℝ) (L : ℝ) : 
  width = 25 ∧ 
  tiles_per_sqft = 4 ∧ 
  green_ratio = 0.4 ∧ 
  red_ratio = 0.6 ∧ 
  green_cost = 3 ∧ 
  red_cost = 1.5 ∧ 
  total_cost = 2100 ∧ 
  total_cost = (green_ratio * tiles_per_sqft * L * width * green_cost) + 
               (red_ratio * tiles_per_sqft * L * width * red_cost) → 
  L = 10 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l376_37642


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l376_37606

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l376_37606


namespace NUMINAMATH_CALUDE_addition_of_decimals_l376_37611

theorem addition_of_decimals : 7.56 + 4.29 = 11.85 := by
  sorry

end NUMINAMATH_CALUDE_addition_of_decimals_l376_37611


namespace NUMINAMATH_CALUDE_coffee_shop_revenue_l376_37682

/-- The number of customers who ordered coffee -/
def coffee_customers : ℕ := 7

/-- The price of a cup of coffee in dollars -/
def coffee_price : ℕ := 5

/-- The number of customers who ordered tea -/
def tea_customers : ℕ := 8

/-- The price of a cup of tea in dollars -/
def tea_price : ℕ := 4

/-- The total revenue of the coffee shop in dollars -/
def total_revenue : ℕ := 67

theorem coffee_shop_revenue :
  coffee_customers * coffee_price + tea_customers * tea_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_coffee_shop_revenue_l376_37682


namespace NUMINAMATH_CALUDE_largest_b_value_l376_37679

theorem largest_b_value (b : ℝ) : 
  (3 * b + 4) * (b - 3) = 9 * b → 
  b ≤ (4 + 4 * Real.sqrt 5) / 6 ∧ 
  ∃ (b : ℝ), (3 * b + 4) * (b - 3) = 9 * b ∧ b = (4 + 4 * Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_b_value_l376_37679


namespace NUMINAMATH_CALUDE_trig_identity_l376_37658

theorem trig_identity : 
  4 * Real.sin (15 * π / 180) + Real.tan (75 * π / 180) = 
  (4 - 3 * (Real.cos (15 * π / 180))^2 + Real.cos (15 * π / 180)) / Real.sin (15 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l376_37658


namespace NUMINAMATH_CALUDE_soda_drinkers_l376_37620

theorem soda_drinkers (total : ℕ) (wine : ℕ) (both : ℕ) (soda : ℕ) : 
  total = 31 → wine = 26 → both = 17 → soda = total + both - wine := by
  sorry

end NUMINAMATH_CALUDE_soda_drinkers_l376_37620


namespace NUMINAMATH_CALUDE_point_upper_left_of_line_l376_37648

/-- A point in the plane is represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane is represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a point is on the upper left side of a line -/
def isUpperLeftSide (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

/-- The main theorem -/
theorem point_upper_left_of_line (t : ℝ) :
  let p : Point := ⟨-2, t⟩
  let l : Line := ⟨1, -1, 4⟩
  isUpperLeftSide p l → t > 2 := by
  sorry


end NUMINAMATH_CALUDE_point_upper_left_of_line_l376_37648


namespace NUMINAMATH_CALUDE_inequality_system_solution_l376_37619

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (5*x - 3 < 3*x + 5 ∧ x < a) ↔ x < 4) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l376_37619


namespace NUMINAMATH_CALUDE_solve_equation_l376_37625

theorem solve_equation (x : ℝ) : 3 * x = (20 - x) + 20 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l376_37625


namespace NUMINAMATH_CALUDE_second_number_proof_l376_37692

theorem second_number_proof : ∃! x : ℤ, 22030 = (555 + x) * (2 * (x - 555)) + 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l376_37692


namespace NUMINAMATH_CALUDE_max_sum_under_constraint_l376_37623

theorem max_sum_under_constraint (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  16 * x * y * z = (x + y)^2 * (x + z)^2 →
  x + y + z ≤ 4 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    16 * a * b * c = (a + b)^2 * (a + c)^2 ∧ a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_under_constraint_l376_37623


namespace NUMINAMATH_CALUDE_first_four_digits_after_decimal_l376_37603

theorem first_four_digits_after_decimal (x : ℝ) : 
  x = (5^1001 + 2)^(5/3) → 
  ∃ n : ℕ, 0 ≤ n ∧ n < 10000 ∧ (x - ⌊x⌋) * 10000 = 3333 + n / 10000 :=
sorry

end NUMINAMATH_CALUDE_first_four_digits_after_decimal_l376_37603


namespace NUMINAMATH_CALUDE_exactly_four_valid_labelings_l376_37656

/-- Represents a truncated 3x3 chessboard with 8 squares. -/
structure TruncatedChessboard :=
  (labels : Fin 8 → Fin 8)

/-- Checks if two positions on the board are connected (share a vertex). -/
def are_connected (p1 p2 : Fin 8) : Bool :=
  sorry

/-- Checks if a labeling is valid according to the problem rules. -/
def is_valid_labeling (board : TruncatedChessboard) : Prop :=
  (∀ p1 p2 : Fin 8, p1 ≠ p2 → board.labels p1 ≠ board.labels p2) ∧
  (∀ p1 p2 : Fin 8, are_connected p1 p2 → 
    (board.labels p1).val + 1 ≠ (board.labels p2).val ∧
    (board.labels p2).val + 1 ≠ (board.labels p1).val)

/-- The main theorem stating that there are exactly 4 valid labelings. -/
theorem exactly_four_valid_labelings :
  ∃! (valid_labelings : Finset TruncatedChessboard),
    (∀ board ∈ valid_labelings, is_valid_labeling board) ∧
    valid_labelings.card = 4 :=
  sorry

end NUMINAMATH_CALUDE_exactly_four_valid_labelings_l376_37656


namespace NUMINAMATH_CALUDE_constant_sequence_l376_37635

def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ (2 * m * n : ℝ) / ((m^2 + n^2) : ℝ)

theorem constant_sequence (a : ℕ → ℝ) (h : sequence_condition a) : ∀ n : ℕ, n > 0 → a n = 1 :=
sorry

end NUMINAMATH_CALUDE_constant_sequence_l376_37635


namespace NUMINAMATH_CALUDE_milk_delivery_calculation_l376_37621

/-- Given a total amount of milk and a difference between two people's deliveries,
    calculate the amount delivered by the person delivering more milk. -/
theorem milk_delivery_calculation (total : ℕ) (difference : ℕ) (h1 : total = 2100) (h2 : difference = 200) :
  (total + difference) / 2 = 1150 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_calculation_l376_37621


namespace NUMINAMATH_CALUDE_garden_area_difference_l376_37662

/-- Represents a rectangular garden with a pathway around it. -/
structure Garden where
  totalLength : ℕ
  totalWidth : ℕ
  pathwayWidth : ℕ

/-- Calculates the effective gardening area of a garden. -/
def effectiveArea (g : Garden) : ℕ :=
  (g.totalLength - 2 * g.pathwayWidth) * (g.totalWidth - 2 * g.pathwayWidth)

/-- Karl's garden dimensions -/
def karlGarden : Garden :=
  { totalLength := 30
  , totalWidth := 50
  , pathwayWidth := 2 }

/-- Makenna's garden dimensions -/
def makennaGarden : Garden :=
  { totalLength := 35
  , totalWidth := 55
  , pathwayWidth := 3 }

/-- Theorem stating the difference in effective gardening area -/
theorem garden_area_difference :
  effectiveArea makennaGarden - effectiveArea karlGarden = 225 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_difference_l376_37662


namespace NUMINAMATH_CALUDE_square_ad_perimeter_l376_37622

theorem square_ad_perimeter (side_length : ℝ) (h : side_length = 8) : 
  4 * side_length = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_ad_perimeter_l376_37622


namespace NUMINAMATH_CALUDE_total_score_is_54_l376_37673

/-- The number of players on the basketball team -/
def num_players : ℕ := 8

/-- The points scored by each player -/
def player_scores : Fin num_players → ℕ
  | ⟨0, _⟩ => 7
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 2
  | ⟨3, _⟩ => 11
  | ⟨4, _⟩ => 6
  | ⟨5, _⟩ => 12
  | ⟨6, _⟩ => 1
  | ⟨7, _⟩ => 7

/-- The theorem stating that the sum of all player scores is 54 -/
theorem total_score_is_54 : (Finset.univ.sum player_scores) = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_54_l376_37673


namespace NUMINAMATH_CALUDE_hybrid_one_headlight_percentage_l376_37618

theorem hybrid_one_headlight_percentage
  (total_cars : ℕ)
  (hybrid_percentage : ℚ)
  (full_headlight_hybrids : ℕ)
  (h1 : total_cars = 600)
  (h2 : hybrid_percentage = 60 / 100)
  (h3 : full_headlight_hybrids = 216) :
  let total_hybrids := (total_cars : ℚ) * hybrid_percentage
  let one_headlight_hybrids := total_hybrids - (full_headlight_hybrids : ℚ)
  one_headlight_hybrids / total_hybrids = 40 / 100 := by
sorry

end NUMINAMATH_CALUDE_hybrid_one_headlight_percentage_l376_37618


namespace NUMINAMATH_CALUDE_hyperbola_b_plus_k_l376_37639

/-- Given a hyperbola with asymptotes y = 3x + 6 and y = -3x + 2, passing through (2, 12),
    prove that b + k = (16√2 + 36) / 9, where (y-k)²/a² - (x-h)²/b² = 1 is the standard form. -/
theorem hyperbola_b_plus_k (a b h k : ℝ) : a > 0 → b > 0 →
  (∀ x y, y = 3*x + 6 ∨ y = -3*x + 2) →  -- Asymptotes
  ((12 - k)^2 / a^2) - ((2 - h)^2 / b^2) = 1 →  -- Point (2, 12) satisfies the equation
  (∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) →  -- Standard form
  b + k = (16 * Real.sqrt 2 + 36) / 9 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_b_plus_k_l376_37639


namespace NUMINAMATH_CALUDE_slope_of_line_l376_37676

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) = (-4/7) * (x - 0) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l376_37676


namespace NUMINAMATH_CALUDE_james_has_more_balloons_l376_37607

/-- The number of balloons James has -/
def james_balloons : ℕ := 232

/-- The number of balloons Amy has -/
def amy_balloons : ℕ := 101

/-- The difference in the number of balloons between James and Amy -/
def balloon_difference : ℕ := james_balloons - amy_balloons

theorem james_has_more_balloons : balloon_difference = 131 := by
  sorry

end NUMINAMATH_CALUDE_james_has_more_balloons_l376_37607


namespace NUMINAMATH_CALUDE_exponent_simplification_l376_37613

theorem exponent_simplification (x : ℝ) (hx : x ≠ 0) :
  x^5 * x^7 / x^3 = x^9 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l376_37613


namespace NUMINAMATH_CALUDE_beads_per_bracelet_l376_37612

theorem beads_per_bracelet (num_friends : ℕ) (current_beads : ℕ) (additional_beads : ℕ) : 
  num_friends = 6 → 
  current_beads = 36 → 
  additional_beads = 12 → 
  (current_beads + additional_beads) / num_friends = 8 :=
by sorry

end NUMINAMATH_CALUDE_beads_per_bracelet_l376_37612


namespace NUMINAMATH_CALUDE_max_students_distribution_l376_37699

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 3540) (h2 : pencils = 2860) :
  Nat.gcd pens pencils = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l376_37699
