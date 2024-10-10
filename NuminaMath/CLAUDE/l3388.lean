import Mathlib

namespace point_transformation_l3388_338850

-- Define the point type
def Point := ℝ × ℝ × ℝ

-- Define the transformations
def reflect_yz (p : Point) : Point :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_z_90 (p : Point) : Point :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : Point) : Point :=
  let (x, y, z) := p
  (x, y, -z)

def rotate_x_180 (p : Point) : Point :=
  let (x, y, z) := p
  (x, -y, -z)

def reflect_xz (p : Point) : Point :=
  let (x, y, z) := p
  (x, -y, z)

-- Define the composition of all transformations
def transform (p : Point) : Point :=
  p |> reflect_yz
    |> rotate_z_90
    |> reflect_xy
    |> rotate_x_180
    |> reflect_xz
    |> rotate_z_90

-- Theorem statement
theorem point_transformation :
  transform (2, 2, 2) = (2, 2, -2) := by
  sorry

end point_transformation_l3388_338850


namespace total_rope_length_l3388_338829

/-- The original length of each rope -/
def rope_length : ℝ := 52

/-- The length used from the first rope -/
def used_first : ℝ := 42

/-- The length used from the second rope -/
def used_second : ℝ := 12

theorem total_rope_length :
  (rope_length - used_first) * 4 = rope_length - used_second →
  2 * rope_length = 104 := by
  sorry

end total_rope_length_l3388_338829


namespace quadratic_roots_range_l3388_338883

theorem quadratic_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, (m + 3) * x₁^2 - 4 * m * x₁ + 2 * m - 1 = 0 ∧
                (m + 3) * x₂^2 - 4 * m * x₂ + 2 * m - 1 = 0 ∧
                x₁ * x₂ < 0 ∧
                x₁ < 0 ∧ x₂ > 0 ∧
                abs x₁ > x₂) →
  m > -3 ∧ m < 0 :=
by sorry

end quadratic_roots_range_l3388_338883


namespace ef_is_one_eighth_of_gh_l3388_338805

/-- Given a line segment GH with points E and F on it, prove that EF is 1/8 of GH -/
theorem ef_is_one_eighth_of_gh (G E F H : Real) :
  (E ≥ G) → (F ≥ G) → (H ≥ E) → (H ≥ F) →  -- E and F lie on GH
  (E - G = 3 * (H - E)) →  -- GE = 3EH
  (F - G = 7 * (H - F)) →  -- GF = 7FH
  abs (E - F) = (1/8) * (H - G) := by sorry

end ef_is_one_eighth_of_gh_l3388_338805


namespace base_of_negative_four_cubed_l3388_338871

def power_expression : ℤ → ℕ → ℤ := (·^·)

theorem base_of_negative_four_cubed :
  ∃ (base : ℤ), power_expression base 3 = power_expression (-4) 3 ∧ base = -4 :=
sorry

end base_of_negative_four_cubed_l3388_338871


namespace min_sum_absolute_values_l3388_338809

theorem min_sum_absolute_values : 
  ∃ (x : ℝ), (∀ (y : ℝ), |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ 
  |x - 1| + |x - 2| + |x - 3| = 2 := by
  sorry

end min_sum_absolute_values_l3388_338809


namespace uncle_omar_parking_probability_l3388_338886

/-- The number of parking spaces -/
def total_spaces : ℕ := 18

/-- The number of cars already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent empty spaces needed -/
def needed_spaces : ℕ := 2

/-- The probability of finding the required adjacent empty spaces -/
def parking_probability : ℚ := 16/51

theorem uncle_omar_parking_probability :
  (1 : ℚ) - (Nat.choose (total_spaces - needed_spaces + 1) parked_cars : ℚ) / 
  (Nat.choose total_spaces parked_cars : ℚ) = parking_probability := by
  sorry

end uncle_omar_parking_probability_l3388_338886


namespace lcm_812_3214_l3388_338891

theorem lcm_812_3214 : Nat.lcm 812 3214 = 1303402 := by
  sorry

end lcm_812_3214_l3388_338891


namespace basketball_lineups_l3388_338834

/-- The number of players in the basketball team -/
def team_size : ℕ := 12

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of different starting lineups that can be chosen -/
def num_lineups : ℕ := 95040

/-- Theorem: The number of different starting lineups that can be chosen
    from a team of 12 players for 5 distinct positions is 95,040 -/
theorem basketball_lineups :
  (team_size.factorial) / ((team_size - lineup_size).factorial) = num_lineups := by
  sorry

end basketball_lineups_l3388_338834


namespace simplify_expression_l3388_338876

theorem simplify_expression (a b c : ℝ) (ha : a = 37/5) (hb : b = 5/37) :
  1.6 * (((1/a + 1/b - 2*c/(a*b)) * (a + b + 2*c)) / (1/a^2 + 1/b^2 + 2/(a*b) - 4*c^2/(a^2*b^2))) = 1.6 :=
by sorry

end simplify_expression_l3388_338876


namespace units_digit_sum_base9_l3388_338862

-- Define a function to convert a base-9 number to base-10
def base9ToBase10 (n : ℕ) : ℕ := 
  (n / 10) * 9 + (n % 10)

-- Define a function to get the units digit in base-9
def unitsDigitBase9 (n : ℕ) : ℕ := 
  n % 9

-- Theorem statement
theorem units_digit_sum_base9 :
  unitsDigitBase9 (base9ToBase10 35 + base9ToBase10 47) = 3 := by
  sorry

end units_digit_sum_base9_l3388_338862


namespace ball_pit_problem_l3388_338804

theorem ball_pit_problem (total : ℕ) (red_fraction : ℚ) (blue_fraction : ℚ) : 
  total = 360 →
  red_fraction = 1/4 →
  blue_fraction = 1/5 →
  ∃ (red blue neither : ℕ),
    red = total * red_fraction ∧
    blue = (total - red) * blue_fraction ∧
    neither = total - red - blue ∧
    neither = 216 :=
by sorry

end ball_pit_problem_l3388_338804


namespace product_of_distinct_numbers_l3388_338843

theorem product_of_distinct_numbers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end product_of_distinct_numbers_l3388_338843


namespace complex_expression_equality_l3388_338872

theorem complex_expression_equality : (8 * 5.4 - 0.6 * 10 / 1.2) ^ 2 = 1459.24 := by
  sorry

end complex_expression_equality_l3388_338872


namespace cauchy_schwarz_two_terms_l3388_338899

theorem cauchy_schwarz_two_terms
  (a b x y : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hx : 0 < x)
  (hy : 0 < y) :
  a * x + b * y ≤ Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) := by
  sorry

end cauchy_schwarz_two_terms_l3388_338899


namespace line_intersection_proof_l3388_338823

theorem line_intersection_proof :
  ∃! (x y : ℚ), 5 * x - 3 * y = 17 ∧ 8 * x + 2 * y = 22 ∧ x = 50 / 17 ∧ y = -13 / 17 := by
  sorry

end line_intersection_proof_l3388_338823


namespace chicken_katsu_cost_is_25_l3388_338860

/-- The cost of the chicken katsu given the following conditions:
  - The family ordered a smoky salmon for $40, a black burger for $15, and a chicken katsu.
  - The bill includes a 10% service charge and 5% tip.
  - Mr. Arevalo paid with $100 and received $8 in change.
-/
def chicken_katsu_cost : ℝ :=
  let salmon_cost : ℝ := 40
  let burger_cost : ℝ := 15
  let service_charge_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.05
  let total_paid : ℝ := 100
  let change_received : ℝ := 8
  let total_bill : ℝ := total_paid - change_received
  25

theorem chicken_katsu_cost_is_25 :
  chicken_katsu_cost = 25 := by sorry

end chicken_katsu_cost_is_25_l3388_338860


namespace unique_solution_system_l3388_338827

theorem unique_solution_system (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  2 * x * Real.sqrt (x + 1) - y * (y + 1) = 1 ∧
  2 * y * Real.sqrt (y + 1) - z * (z + 1) = 1 ∧
  2 * z * Real.sqrt (z + 1) - x * (x + 1) = 1 →
  x = (1 + Real.sqrt 5) / 2 ∧
  y = (1 + Real.sqrt 5) / 2 ∧
  z = (1 + Real.sqrt 5) / 2 :=
by sorry

end unique_solution_system_l3388_338827


namespace students_without_eyewear_l3388_338810

/-- Given a population of students with specified percentages wearing glasses, contact lenses, or both,
    calculate the number of students not wearing any eyewear. -/
theorem students_without_eyewear
  (total_students : ℕ)
  (glasses_percent : ℚ)
  (contacts_percent : ℚ)
  (both_percent : ℚ)
  (h_total : total_students = 1200)
  (h_glasses : glasses_percent = 35 / 100)
  (h_contacts : contacts_percent = 25 / 100)
  (h_both : both_percent = 10 / 100) :
  (total_students : ℚ) * (1 - (glasses_percent + contacts_percent - both_percent)) = 600 :=
sorry

end students_without_eyewear_l3388_338810


namespace password_matches_stored_sequence_l3388_338807

/-- Represents a 32-letter alphabet where each letter is encoded as a pair of digits. -/
def Alphabet : Type := Fin 32

/-- Converts a letter to its ordinal number representation. -/
def toOrdinal (a : Alphabet) : Fin 100 := sorry

/-- Represents the remainder when dividing by 10. -/
def r10 (x : ℕ) : Fin 10 := sorry

/-- Generates the x_i sequence based on the given recurrence relation. -/
def genX (a b : ℕ) : ℕ → Fin 10
  | 0 => sorry
  | n + 1 => r10 (a * (genX a b n).val + b)

/-- Generates the c_i sequence based on x_i and y_i. -/
def genC (x : ℕ → Fin 10) (y : ℕ → Fin 100) : ℕ → Fin 10 :=
  fun i => r10 (x i + (y i).val)

/-- Converts a string to a sequence of ordinal numbers. -/
def stringToOrdinals (s : String) : ℕ → Fin 100 := sorry

/-- The stored sequence c_i. -/
def storedSequence : List (Fin 10) :=
  [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7, 5]

/-- The password in lowercase letters. -/
def password : String := "яхта"

theorem password_matches_stored_sequence :
  ∃ (a b : ℕ),
    (∀ i, i ≥ 10 → genX a b i = genX a b (i - 10)) ∧
    genC (genX a b) (stringToOrdinals (password ++ password)) = fun i =>
      if h : i < storedSequence.length then
        storedSequence[i]'h
      else
        0 := by sorry

end password_matches_stored_sequence_l3388_338807


namespace inequalities_proof_l3388_338885

theorem inequalities_proof (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
sorry

end inequalities_proof_l3388_338885


namespace acetone_molecular_weight_proof_l3388_338897

-- Define the isotopes and their properties
structure Isotope where
  mass : Float
  abundance : Float

-- Define the elements and their isotopes
def carbon_isotopes : List Isotope := [
  { mass := 12, abundance := 0.9893 },
  { mass := 13.003355, abundance := 0.0107 }
]

def hydrogen_isotopes : List Isotope := [
  { mass := 1.007825, abundance := 0.999885 },
  { mass := 2.014102, abundance := 0.000115 }
]

def oxygen_isotopes : List Isotope := [
  { mass := 15.994915, abundance := 0.99757 },
  { mass := 16.999132, abundance := 0.00038 },
  { mass := 17.999159, abundance := 0.00205 }
]

-- Function to calculate average atomic mass
def average_atomic_mass (isotopes : List Isotope) : Float :=
  isotopes.foldl (fun acc isotope => acc + isotope.mass * isotope.abundance) 0

-- Define the molecular formula of Acetone
def acetone_formula : List (Nat × List Isotope) := [
  (3, carbon_isotopes),
  (6, hydrogen_isotopes),
  (1, oxygen_isotopes)
]

-- Calculate the molecular weight of Acetone
def acetone_molecular_weight : Float :=
  acetone_formula.foldl (fun acc (n, isotopes) => acc + n.toFloat * average_atomic_mass isotopes) 0

-- Theorem statement
theorem acetone_molecular_weight_proof :
  (acetone_molecular_weight - 58.107055).abs < 0.000001 := by
  sorry


end acetone_molecular_weight_proof_l3388_338897


namespace negative_sum_l3388_338870

theorem negative_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  b + c < 0 := by
sorry

end negative_sum_l3388_338870


namespace bakery_tart_flour_calculation_l3388_338863

theorem bakery_tart_flour_calculation 
  (initial_tarts : ℕ) 
  (new_tarts : ℕ) 
  (initial_flour_per_tart : ℚ) 
  (h1 : initial_tarts = 36)
  (h2 : new_tarts = 18)
  (h3 : initial_flour_per_tart = 1 / 12)
  : (initial_tarts : ℚ) * initial_flour_per_tart / new_tarts = 1 / 6 := by
  sorry

end bakery_tart_flour_calculation_l3388_338863


namespace two_numbers_problem_l3388_338867

theorem two_numbers_problem (A B : ℝ) (h1 : A + B = 40) (h2 : A * B = 375) (h3 : A / B = 3/2) 
  (h4 : A > 0) (h5 : B > 0) : A = 24 ∧ B = 16 ∧ A - B = 8 := by
  sorry

end two_numbers_problem_l3388_338867


namespace inequality_proof_l3388_338812

theorem inequality_proof (a b c : ℕ) (ha : a = 8^53) (hb : b = 16^41) (hc : c = 64^27) :
  b > c ∧ c > a := by
  sorry

end inequality_proof_l3388_338812


namespace cubic_sum_l3388_338802

theorem cubic_sum (a b c : ℝ) 
  (h1 : a + b + c = 8) 
  (h2 : a * b + a * c + b * c = 9) 
  (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 242 := by
  sorry

end cubic_sum_l3388_338802


namespace min_distance_to_line_l3388_338898

theorem min_distance_to_line : 
  ∃ (d : ℝ), d > 0 ∧ 
  (∀ a b : ℝ, a + 2*b = Real.sqrt 5 → Real.sqrt (a^2 + b^2) ≥ d) ∧
  (∃ a b : ℝ, a + 2*b = Real.sqrt 5 ∧ Real.sqrt (a^2 + b^2) = d) ∧
  d = 1 := by
sorry

end min_distance_to_line_l3388_338898


namespace lcm_72_98_l3388_338859

theorem lcm_72_98 : Nat.lcm 72 98 = 3528 := by
  sorry

end lcm_72_98_l3388_338859


namespace largest_divisible_by_six_under_9000_l3388_338875

theorem largest_divisible_by_six_under_9000 : 
  ∀ n : ℕ, n < 9000 ∧ 6 ∣ n → n ≤ 8994 :=
by
  sorry

end largest_divisible_by_six_under_9000_l3388_338875


namespace intersection_implies_values_l3388_338821

/-- Sets T and S in the xy-plane -/
def T (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + p.2 - 3 = 0}
def S (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - b = 0}

/-- The main theorem -/
theorem intersection_implies_values (a b : ℝ) :
  T a ∩ S b = {(2, 1)} → a = 1 ∧ b = 1 := by
  sorry

end intersection_implies_values_l3388_338821


namespace candy_ratio_l3388_338861

theorem candy_ratio (adam james rubert : ℕ) : 
  adam = 6 →
  james = 3 * adam →
  adam + james + rubert = 96 →
  rubert = 4 * james :=
by
  sorry

end candy_ratio_l3388_338861


namespace liam_paid_more_than_ellen_l3388_338830

-- Define the pizza characteristics
def total_slices : ℕ := 12
def plain_pizza_cost : ℚ := 12
def extra_cheese_cost : ℚ := 3
def extra_cheese_slices : ℕ := total_slices / 3

-- Define what Liam and Ellen ate
def liam_extra_cheese_slices : ℕ := extra_cheese_slices
def liam_plain_slices : ℕ := 4
def ellen_plain_slices : ℕ := total_slices - liam_extra_cheese_slices - liam_plain_slices

-- Calculate total pizza cost
def total_pizza_cost : ℚ := plain_pizza_cost + extra_cheese_cost

-- Calculate cost per slice
def cost_per_slice : ℚ := total_pizza_cost / total_slices

-- Calculate what Liam and Ellen paid
def liam_payment : ℚ := cost_per_slice * (liam_extra_cheese_slices + liam_plain_slices)
def ellen_payment : ℚ := (plain_pizza_cost / total_slices) * ellen_plain_slices

-- Theorem to prove
theorem liam_paid_more_than_ellen : liam_payment - ellen_payment = 6 := by
  sorry

end liam_paid_more_than_ellen_l3388_338830


namespace equation_solution_l3388_338817

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7))
  ∀ x : ℝ, f x = 1/8 ↔ x = 4 + Real.sqrt 57 ∨ x = 4 - Real.sqrt 57 :=
by sorry

end equation_solution_l3388_338817


namespace sock_selection_l3388_338892

theorem sock_selection (n k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  Nat.choose n k = 15 := by
  sorry

end sock_selection_l3388_338892


namespace cylinder_volume_l3388_338838

/-- Given a cylinder with height 2 and lateral surface area 4π, its volume is 2π -/
theorem cylinder_volume (h : ℝ) (lateral_area : ℝ) (volume : ℝ) : 
  h = 2 → lateral_area = 4 * Real.pi → volume = 2 * Real.pi :=
by sorry

end cylinder_volume_l3388_338838


namespace quartic_root_sum_l3388_338864

/-- Given a quartic equation px^4 + qx^3 + rx^2 + sx + t = 0 with roots 4, -3, and 0, 
    and p ≠ 0, prove that (q+r)/p = -13 -/
theorem quartic_root_sum (p q r s t : ℝ) (hp : p ≠ 0) 
  (h1 : p * 4^4 + q * 4^3 + r * 4^2 + s * 4 + t = 0)
  (h2 : p * (-3)^4 + q * (-3)^3 + r * (-3)^2 + s * (-3) + t = 0)
  (h3 : t = 0) : 
  (q + r) / p = -13 := by
  sorry

end quartic_root_sum_l3388_338864


namespace min_radius_point_l3388_338819

/-- The point that minimizes the radius of a circle centered at the origin -/
theorem min_radius_point (x y : ℝ) :
  (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x = 0 ∧ y = 0 := by
  sorry

end min_radius_point_l3388_338819


namespace track_extension_calculation_l3388_338893

/-- Theorem: Track Extension Calculation
Given a train track with an elevation gain of 600 meters,
changing the gradient from 3% to 2% results in a track extension of 10 km. -/
theorem track_extension_calculation (elevation_gain : ℝ) (initial_gradient : ℝ) (final_gradient : ℝ) :
  elevation_gain = 600 →
  initial_gradient = 0.03 →
  final_gradient = 0.02 →
  (elevation_gain / final_gradient - elevation_gain / initial_gradient) / 1000 = 10 := by
  sorry

#check track_extension_calculation

end track_extension_calculation_l3388_338893


namespace coefficient_sum_l3388_338806

variables (a b c d e : ℝ)

/-- The polynomial equation -/
def polynomial (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- Theorem stating the relationship between the coefficients of the polynomial -/
theorem coefficient_sum (h1 : a ≠ 0)
  (h2 : polynomial 5 = 0)
  (h3 : polynomial (-3) = 0)
  (h4 : polynomial 1 = 0) :
  (b + c + d) / a = -7 := by sorry

end coefficient_sum_l3388_338806


namespace division_multiplication_result_l3388_338896

theorem division_multiplication_result : (-6) / (-6) * (-1/6 : ℚ) = -1/6 := by
  sorry

end division_multiplication_result_l3388_338896


namespace merchant_loss_l3388_338800

theorem merchant_loss (C S : ℝ) (h : C > 0) :
  40 * C = 25 * S → (S - C) / C * 100 = -20 := by
  sorry

end merchant_loss_l3388_338800


namespace ages_sum_l3388_338895

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
sorry

end ages_sum_l3388_338895


namespace num_plane_determining_pairs_eq_66_l3388_338816

/-- A rectangular prism with distinct dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  distinct : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of edges in a rectangular prism -/
def num_edges : ℕ := 12

/-- The number of unordered pairs of parallel edges -/
def num_parallel_pairs : ℕ := 18

/-- The total number of unordered pairs of edges -/
def total_edge_pairs : ℕ := num_edges * (num_edges - 1) / 2

/-- The number of unordered pairs of edges that determine a plane -/
def num_plane_determining_pairs (prism : RectangularPrism) : ℕ :=
  total_edge_pairs

/-- Theorem: The number of unordered pairs of edges in a rectangular prism
    with distinct dimensions that determine a plane is 66 -/
theorem num_plane_determining_pairs_eq_66 (prism : RectangularPrism) :
  num_plane_determining_pairs prism = 66 := by
  sorry

end num_plane_determining_pairs_eq_66_l3388_338816


namespace class_average_score_l3388_338879

theorem class_average_score (scores : List ℝ) (avg_others : ℝ) : 
  scores.length = 4 →
  scores = [90, 85, 88, 80] →
  avg_others = 82 →
  let total_students : ℕ := 30
  let sum_scores : ℝ := scores.sum + (total_students - 4 : ℕ) * avg_others
  sum_scores / total_students = 82.5 := by
  sorry

end class_average_score_l3388_338879


namespace trig_equation_roots_l3388_338833

open Real

theorem trig_equation_roots (α β : ℝ) : 
  0 < α ∧ α < π ∧ 0 < β ∧ β < π →
  (∃ x y : ℝ, x^2 - 5*x + 6 = 0 ∧ y^2 - 5*y + 6 = 0 ∧ x = tan α ∧ y = tan β) →
  tan (α + β) = -1 ∧ cos (α - β) = (7 * Real.sqrt 2) / 10 := by
  sorry

end trig_equation_roots_l3388_338833


namespace parallelogram_point_D_l3388_338822

/-- A point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- A parallelogram in the complex plane -/
structure Parallelogram where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- The given parallelogram ABCD -/
def givenParallelogram : Parallelogram where
  A := { re := 4, im := 1 }
  B := { re := 3, im := 4 }
  C := { re := 5, im := 2 }
  D := { re := 6, im := -1 }

theorem parallelogram_point_D (p : Parallelogram) (h : p = givenParallelogram) :
  p.D.re = 6 ∧ p.D.im = -1 := by
  sorry

end parallelogram_point_D_l3388_338822


namespace triangle_arctan_sum_l3388_338856

theorem triangle_arctan_sum (a b c : ℝ) (h : c = a + b) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.arctan (1 / 2) :=
by sorry

end triangle_arctan_sum_l3388_338856


namespace sum_of_coordinates_after_reflection_l3388_338814

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let A : ℝ × ℝ := (x, 8)
  let B : ℝ × ℝ := reflect_over_y_axis A
  A.1 + A.2 + B.1 + B.2 = 16 := by sorry

end sum_of_coordinates_after_reflection_l3388_338814


namespace square_inequality_not_sufficient_nor_necessary_l3388_338828

theorem square_inequality_not_sufficient_nor_necessary (x y : ℝ) :
  ¬(∀ x y : ℝ, x^2 > y^2 → x > y) ∧ ¬(∀ x y : ℝ, x > y → x^2 > y^2) := by
  sorry

end square_inequality_not_sufficient_nor_necessary_l3388_338828


namespace arithmetic_geometric_sequence_l3388_338849

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →  -- non-zero common difference
  a 1 = 1 →  -- a_1 = 1
  (a 2) * (a 5) = (a 4)^2 →  -- a_2, a_4, and a_5 form a geometric sequence
  d = 1/5 := by sorry

end arithmetic_geometric_sequence_l3388_338849


namespace sqrt_equation_average_zero_l3388_338880

theorem sqrt_equation_average_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 40}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
  ∀ x ∈ solutions, x = x₁ ∨ x = x₂ ∧
  (x₁ + x₂) / 2 = 0 :=
by sorry

end sqrt_equation_average_zero_l3388_338880


namespace money_sharing_l3388_338847

theorem money_sharing (emani howard jamal : ℕ) (h1 : emani = 150) (h2 : emani = howard + 30) (h3 : jamal = 75) :
  (emani + howard + jamal) / 3 = 115 := by
  sorry

end money_sharing_l3388_338847


namespace rectangle_width_l3388_338889

/-- Proves that the width of a rectangle is 5 cm, given the specified conditions -/
theorem rectangle_width (length width : ℝ) : 
  (2 * length + 2 * width = 16) →  -- Perimeter is 16 cm
  (width = length + 2) →           -- Width is 2 cm longer than length
  width = 5 := by
sorry


end rectangle_width_l3388_338889


namespace intersection_A_B_range_of_a_l3388_338813

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x+2)*(4-x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a+1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∪ C a = B → a ∈ Set.Icc (-2) 3 := by sorry

end intersection_A_B_range_of_a_l3388_338813


namespace negation_of_union_membership_l3388_338801

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end negation_of_union_membership_l3388_338801


namespace floor_ceiling_sum_l3388_338837

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by sorry

end floor_ceiling_sum_l3388_338837


namespace equation_roots_l3388_338835

theorem equation_roots (a b : ℝ) :
  -- Part 1
  (∃ x : ℂ, x = 1 - Complex.I * Real.sqrt 3 ∧ x / a + b / x = 1) →
  a = 2 ∧ b = 2
  ∧
  -- Part 2
  (b / a > 1 / 4 ∧ a > 0) →
  ¬∃ x : ℝ, x / a + b / x = 1 :=
by sorry

end equation_roots_l3388_338835


namespace hexagonal_table_dice_probability_l3388_338888

/-- The number of people seated around the hexagonal table -/
def num_people : ℕ := 6

/-- The number of sides on the standard die -/
def die_sides : ℕ := 6

/-- A function to calculate the number of valid options for each person's roll -/
def valid_options (person : ℕ) : ℕ :=
  match person with
  | 1 => 6  -- Person A
  | 2 => 5  -- Person B
  | 3 => 4  -- Person C
  | 4 => 5  -- Person D
  | 5 => 3  -- Person E
  | 6 => 3  -- Person F
  | _ => 0  -- Invalid person number

/-- The probability of no two adjacent or opposite people rolling the same number -/
def probability : ℚ :=
  (valid_options 1 * valid_options 2 * valid_options 3 * valid_options 4 * valid_options 5 * valid_options 6) /
  (die_sides ^ num_people)

theorem hexagonal_table_dice_probability :
  probability = 25 / 648 := by
  sorry

end hexagonal_table_dice_probability_l3388_338888


namespace not_all_fractions_integer_l3388_338840

theorem not_all_fractions_integer 
  (a b c r s t : ℕ+) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (eq1 : a * b + 1 = r^2)
  (eq2 : a * c + 1 = s^2)
  (eq3 : b * c + 1 = t^2) :
  ¬(∃ (x y z : ℕ), (r * t : ℚ) / s = x ∧ (r * s : ℚ) / t = y ∧ (s * t : ℚ) / r = z) :=
sorry

end not_all_fractions_integer_l3388_338840


namespace three_operations_to_one_tile_l3388_338832

/-- Represents the set of tiles -/
def TileSet := Finset Nat

/-- The operation of removing perfect squares and renumbering -/
def remove_squares_and_renumber (s : TileSet) : TileSet :=
  sorry

/-- The initial set of tiles from 1 to 49 -/
def initial_set : TileSet :=
  sorry

/-- Applies the operation n times -/
def apply_n_times (n : Nat) (s : TileSet) : TileSet :=
  sorry

theorem three_operations_to_one_tile :
  ∃ (n : Nat), n = 3 ∧ (apply_n_times n initial_set).card = 1 ∧
  ∀ (m : Nat), m < n → (apply_n_times m initial_set).card > 1 :=
sorry

end three_operations_to_one_tile_l3388_338832


namespace circle_area_from_points_l3388_338882

/-- The area of a circle with diameter endpoints at A(-1,3) and B'(13,12) is 277π/4 square units. -/
theorem circle_area_from_points :
  let A : ℝ × ℝ := (-1, 3)
  let B' : ℝ × ℝ := (13, 12)
  let diameter := Real.sqrt ((B'.1 - A.1)^2 + (B'.2 - A.2)^2)
  let radius := diameter / 2
  let area := π * radius^2
  area = 277 * π / 4 := by
sorry

end circle_area_from_points_l3388_338882


namespace sqrt_negative_undefined_l3388_338826

theorem sqrt_negative_undefined : ¬ ∃ (x : ℝ), x^2 = -1/4 := by
  sorry

end sqrt_negative_undefined_l3388_338826


namespace repair_easier_than_thermometer_l3388_338878

def word1 : String := "термометр"
def word2 : String := "ремонт"

def uniqueLetters (s : String) : Finset Char :=
  s.toList.toFinset

theorem repair_easier_than_thermometer :
  (uniqueLetters word2).card > (uniqueLetters word1).card := by
  sorry

end repair_easier_than_thermometer_l3388_338878


namespace snow_volume_calculation_l3388_338854

/-- The volume of snow to be shoveled from a walkway -/
def snow_volume (total_length width depth no_shovel_length : ℝ) : ℝ :=
  (total_length - no_shovel_length) * width * depth

/-- Proof that the volume of snow to be shoveled is 46.875 cubic feet -/
theorem snow_volume_calculation : 
  snow_volume 30 2.5 0.75 5 = 46.875 := by
  sorry

end snow_volume_calculation_l3388_338854


namespace old_toilet_water_usage_l3388_338836

/-- The amount of water saved by switching to a new toilet in June -/
def water_saved : ℝ := 1800

/-- The number of times the toilet is flushed per day -/
def flushes_per_day : ℕ := 15

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The percentage of water saved by the new toilet compared to the old one -/
def water_saving_percentage : ℝ := 0.8

theorem old_toilet_water_usage : ℝ :=
  let total_flushes : ℕ := flushes_per_day * days_in_june
  let water_saved_per_flush : ℝ := water_saved / total_flushes
  water_saved_per_flush / water_saving_percentage

#check @old_toilet_water_usage

end old_toilet_water_usage_l3388_338836


namespace total_distance_is_55km_l3388_338851

/-- Represents the distances Ivan ran on each day of the week -/
structure RunningDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The conditions of Ivan's running schedule -/
def validRunningSchedule (d : RunningDistances) : Prop :=
  d.tuesday = 2 * d.monday ∧
  d.wednesday = d.tuesday / 2 ∧
  d.thursday = d.wednesday / 2 ∧
  d.friday = 2 * d.thursday ∧
  d.thursday = 5 -- The shortest distance is 5 km, which occurs on Thursday

/-- The theorem to prove -/
theorem total_distance_is_55km (d : RunningDistances) 
  (h : validRunningSchedule d) : 
  d.monday + d.tuesday + d.wednesday + d.thursday + d.friday = 55 := by
  sorry


end total_distance_is_55km_l3388_338851


namespace mike_peaches_picked_l3388_338831

/-- The number of peaches Mike picked -/
def peaches_picked (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem mike_peaches_picked : 
  peaches_picked 34 86 = 52 := by sorry

end mike_peaches_picked_l3388_338831


namespace certain_number_proof_l3388_338868

theorem certain_number_proof (n : ℕ) (h1 : n > 0) :
  let m := 72 * 14
  Nat.gcd m 72 = 72 :=
by sorry

end certain_number_proof_l3388_338868


namespace no_perfect_squares_l3388_338866

theorem no_perfect_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n.val^2 - 1 = a^2) ∧ (3 * n.val^2 - 1 = b^2) ∧ (6 * n.val^2 - 1 = c^2)) :=
by sorry

end no_perfect_squares_l3388_338866


namespace alice_win_condition_l3388_338873

/-- The game state represents the positions of the red and blue pieces -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- Alice's move function -/
def move (r : ℚ) (state : GameState) (k : ℤ) : GameState :=
  { red := state.red,
    blue := state.red + r^k * (state.blue - state.red) }

/-- Alice can win the game -/
def can_win (r : ℚ) : Prop :=
  ∃ (moves : List ℤ), moves.length ≤ 2021 ∧
    (moves.foldl (move r) { red := 0, blue := 1 }).red = 1

/-- The main theorem stating the condition for Alice to win -/
theorem alice_win_condition (r : ℚ) : 
  (r > 1 ∧ can_win r) ↔ (∃ d : ℕ, d ≥ 1 ∧ d ≤ 1010 ∧ r = 1 + 1 / d) := by
  sorry


end alice_win_condition_l3388_338873


namespace trapezoid_longer_side_length_l3388_338825

-- Define the square
def square_side_length : ℝ := 2

-- Define the number of regions the square is divided into
def num_regions : ℕ := 3

-- Define the theorem
theorem trapezoid_longer_side_length :
  ∀ (trapezoid_area pentagon_area : ℝ),
  trapezoid_area > 0 →
  pentagon_area > 0 →
  trapezoid_area = pentagon_area →
  trapezoid_area = (square_side_length ^ 2) / num_regions →
  ∃ (y : ℝ),
    y = 5 / 3 ∧
    trapezoid_area = (y + 1) / 2 := by
  sorry


end trapezoid_longer_side_length_l3388_338825


namespace chef_michel_pies_l3388_338841

/-- Calculates the total number of pies sold given the number of slices per pie and the number of slices ordered --/
def total_pies_sold (shepherds_pie_slices : ℕ) (chicken_pot_pie_slices : ℕ) 
                    (shepherds_pie_ordered : ℕ) (chicken_pot_pie_ordered : ℕ) : ℕ :=
  (shepherds_pie_ordered / shepherds_pie_slices) + (chicken_pot_pie_ordered / chicken_pot_pie_slices)

/-- Proves that Chef Michel sold 29 pies in total --/
theorem chef_michel_pies : 
  total_pies_sold 4 5 52 80 = 29 := by
  sorry

end chef_michel_pies_l3388_338841


namespace tangency_implies_n_equals_two_l3388_338824

/-- The value of n for which the ellipse x^2 + 9y^2 = 9 is tangent to the hyperbola x^2 - n(y - 1)^2 = 1 -/
def tangency_value : ℝ := 2

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := x^2 - n*(y - 1)^2 = 1

/-- The ellipse and hyperbola are tangent -/
def are_tangent (n : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_ellipse x y ∧ is_on_hyperbola x y n ∧
  ∀ x' y' : ℝ, is_on_ellipse x' y' ∧ is_on_hyperbola x' y' n → (x', y') = (x, y)

theorem tangency_implies_n_equals_two :
  are_tangent tangency_value := by sorry

end tangency_implies_n_equals_two_l3388_338824


namespace max_value_abc_max_value_abc_achievable_l3388_338844

theorem max_value_abc (A B C : ℕ) (h : A + B + C = 15) :
  (A * B * C + A * B + B * C + C * A) ≤ 200 :=
by sorry

theorem max_value_abc_achievable :
  ∃ (A B C : ℕ), A + B + C = 15 ∧ A * B * C + A * B + B * C + C * A = 200 :=
by sorry

end max_value_abc_max_value_abc_achievable_l3388_338844


namespace f_range_theorem_l3388_338803

noncomputable def f (x : ℝ) : ℝ := x + (Real.exp x)⁻¹

theorem f_range_theorem :
  {a : ℝ | ∀ x, f x > a * x} = Set.Ioo (1 - Real.exp 1) 1 :=
sorry

end f_range_theorem_l3388_338803


namespace cost_of_ingredients_for_two_cakes_l3388_338877

/-- The cost of ingredients for two cakes given selling price, profit, and packaging cost -/
theorem cost_of_ingredients_for_two_cakes 
  (selling_price : ℝ) 
  (profit_per_cake : ℝ) 
  (packaging_cost : ℝ) : 
  selling_price = 15 → 
  profit_per_cake = 8 → 
  packaging_cost = 1 → 
  2 * selling_price - 2 * profit_per_cake - 2 * packaging_cost = 12 := by
  sorry

end cost_of_ingredients_for_two_cakes_l3388_338877


namespace zoe_recycled_pounds_l3388_338852

/-- The number of pounds that earn one point -/
def pounds_per_point : ℕ := 8

/-- The number of pounds Zoe's friends recycled -/
def friends_pounds : ℕ := 23

/-- The total number of points earned -/
def total_points : ℕ := 6

/-- The number of pounds Zoe recycled -/
def zoe_pounds : ℕ := 25

theorem zoe_recycled_pounds :
  zoe_pounds + friends_pounds = pounds_per_point * total_points :=
sorry

end zoe_recycled_pounds_l3388_338852


namespace choir_size_choir_size_is_30_l3388_338815

/-- The number of singers in a school choir, given the initial number of robes,
    the cost per new robe, and the total amount spent on new robes. -/
theorem choir_size (initial_robes : ℕ) (cost_per_robe : ℕ) (total_spent : ℕ) : ℕ :=
  initial_robes + total_spent / cost_per_robe

/-- Proof that the number of singers in the choir is 30. -/
theorem choir_size_is_30 :
  choir_size 12 2 36 = 30 := by
  sorry

end choir_size_choir_size_is_30_l3388_338815


namespace wombat_count_l3388_338848

/-- The number of wombats in the enclosure -/
def num_wombats : ℕ := 9

/-- The number of rheas in the enclosure -/
def num_rheas : ℕ := 3

/-- The number of times each wombat claws -/
def wombat_claws : ℕ := 4

/-- The number of times each rhea claws -/
def rhea_claws : ℕ := 1

/-- The total number of claws -/
def total_claws : ℕ := 39

theorem wombat_count : 
  num_wombats * wombat_claws + num_rheas * rhea_claws = total_claws :=
by sorry

end wombat_count_l3388_338848


namespace sarah_initial_cupcakes_l3388_338853

/-- The number of cupcakes Todd ate -/
def cupcakes_eaten : ℕ := 14

/-- The number of packages Sarah could make after Todd ate some cupcakes -/
def packages : ℕ := 3

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 8

/-- The initial number of cupcakes Sarah baked -/
def initial_cupcakes : ℕ := cupcakes_eaten + packages * cupcakes_per_package

theorem sarah_initial_cupcakes : initial_cupcakes = 38 := by
  sorry

end sarah_initial_cupcakes_l3388_338853


namespace danny_steve_time_ratio_l3388_338845

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := 29

/-- The time it takes Steve to reach Danny's house -/
def steve_time : ℝ := 58

/-- The difference in time it takes Steve and Danny to reach the halfway point -/
def halfway_time_difference : ℝ := 14.5

theorem danny_steve_time_ratio :
  danny_time / steve_time = 1 / 2 ∧
  steve_time / 2 = danny_time / 2 + halfway_time_difference :=
by sorry

end danny_steve_time_ratio_l3388_338845


namespace final_number_is_81_l3388_338869

/-- Represents the elimination process on a list of numbers -/
def elimination_process (n : ℕ) : ℕ :=
  if n ≤ 3 then n else
  let m := elimination_process ((2 * n + 3) / 3)
  if m * 3 > n then m else m + 1

/-- The theorem stating that 81 is the final remaining number -/
theorem final_number_is_81 : elimination_process 200 = 81 := by
  sorry

end final_number_is_81_l3388_338869


namespace coin_flip_probability_l3388_338865

theorem coin_flip_probability (n : ℕ) (k : ℕ) (h : n = 4 ∧ k = 3) :
  (2 : ℚ) / (2^n : ℚ) = 1/8 := by sorry

end coin_flip_probability_l3388_338865


namespace john_gave_money_to_two_friends_l3388_338811

/-- Calculates the number of friends John gave money to -/
def number_of_friends (initial_amount spent_on_sweets amount_per_friend final_amount : ℚ) : ℕ :=
  ((initial_amount - spent_on_sweets - final_amount) / amount_per_friend).num.toNat

/-- Proves that John gave money to 2 friends -/
theorem john_gave_money_to_two_friends :
  number_of_friends 7.10 1.05 1.00 4.05 = 2 := by
  sorry

#eval number_of_friends 7.10 1.05 1.00 4.05

end john_gave_money_to_two_friends_l3388_338811


namespace darias_current_money_l3388_338855

/-- Calculates Daria's current money for concert tickets -/
theorem darias_current_money
  (ticket_cost : ℕ)  -- Cost of one ticket
  (num_tickets : ℕ)  -- Number of tickets Daria needs to buy
  (money_needed : ℕ) -- Additional money Daria needs to earn
  (h1 : ticket_cost = 90)
  (h2 : num_tickets = 4)
  (h3 : money_needed = 171) :
  ticket_cost * num_tickets - money_needed = 189 :=
by sorry

end darias_current_money_l3388_338855


namespace four_digit_divisible_by_45_l3388_338820

-- Define a function to represent a four-digit number of the form a43b
def number (a b : Nat) : Nat := a * 1000 + 430 + b

-- Define the divisibility condition
def isDivisibleBy45 (n : Nat) : Prop := n % 45 = 0

-- State the theorem
theorem four_digit_divisible_by_45 :
  ∀ a b : Nat, a < 10 ∧ b < 10 →
    (isDivisibleBy45 (number a b) ↔ (a = 2 ∧ b = 0) ∨ (a = 6 ∧ b = 5)) := by
  sorry

end four_digit_divisible_by_45_l3388_338820


namespace library_problem_l3388_338842

/-- Calculates the number of students helped on the first day given the total books,
    books per student, and students helped on subsequent days. -/
def students_helped_first_day (total_books : ℕ) (books_per_student : ℕ) 
    (students_day2 : ℕ) (students_day3 : ℕ) (students_day4 : ℕ) : ℕ :=
  (total_books - (students_day2 + students_day3 + students_day4) * books_per_student) / books_per_student

/-- Theorem stating that given the conditions in the problem, 
    the number of students helped on the first day is 4. -/
theorem library_problem : 
  students_helped_first_day 120 5 5 6 9 = 4 := by
  sorry

end library_problem_l3388_338842


namespace same_solution_implies_k_equals_one_l3388_338874

theorem same_solution_implies_k_equals_one :
  ∀ k : ℝ,
  (∀ x : ℝ, 4*x + 3*k = 2*x + 2 ↔ 2*x + k = 5*x + 2.5) →
  k = 1 := by
  sorry

end same_solution_implies_k_equals_one_l3388_338874


namespace equation_solution_l3388_338884

theorem equation_solution : 
  let x : ℚ := -7/6
  (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 := by sorry

end equation_solution_l3388_338884


namespace candy_distribution_l3388_338881

theorem candy_distribution (n : ℕ) : 
  n > 0 → 
  (∃ k : ℕ, n * k + 1 = 120) → 
  n = 7 ∨ n = 17 :=
sorry

end candy_distribution_l3388_338881


namespace pyramid_min_faces_l3388_338890

/-- A pyramid is a three-dimensional polyhedron with a polygonal base and triangular faces meeting at a common point (apex). -/
structure Pyramid where
  faces : ℕ

/-- Theorem: The number of faces in any pyramid is at least 4. -/
theorem pyramid_min_faces (p : Pyramid) : p.faces ≥ 4 := by
  sorry

end pyramid_min_faces_l3388_338890


namespace sum_even_102_to_200_proof_l3388_338846

/-- The sum of even integers from 102 to 200 inclusive -/
def sum_even_102_to_200 : ℕ := 7550

/-- The sum of the first 50 positive even integers -/
def sum_first_50_even : ℕ := 2550

/-- The number of even integers from 102 to 200 inclusive -/
def num_even_102_to_200 : ℕ := 50

/-- The first even integer in the range 102 to 200 -/
def first_even_102_to_200 : ℕ := 102

/-- The last even integer in the range 102 to 200 -/
def last_even_102_to_200 : ℕ := 200

theorem sum_even_102_to_200_proof :
  sum_even_102_to_200 = (num_even_102_to_200 / 2) * (first_even_102_to_200 + last_even_102_to_200) :=
by sorry

end sum_even_102_to_200_proof_l3388_338846


namespace correct_ingredients_l3388_338894

/-- Recipe proportions and banana usage --/
structure RecipeData where
  flour_to_mush : ℚ  -- ratio of flour to banana mush
  sugar_to_mush : ℚ  -- ratio of sugar to banana mush
  milk_to_flour : ℚ  -- ratio of milk to flour
  bananas_per_mush : ℕ  -- number of bananas per cup of mush
  total_bananas : ℕ  -- total number of bananas used

/-- Calculated ingredients based on recipe data --/
def calculate_ingredients (r : RecipeData) : ℚ × ℚ × ℚ :=
  let mush := r.total_bananas / r.bananas_per_mush
  let flour := mush * r.flour_to_mush
  let sugar := mush * r.sugar_to_mush
  let milk := flour * r.milk_to_flour
  (flour, sugar, milk)

/-- Theorem stating the correct amounts of ingredients --/
theorem correct_ingredients (r : RecipeData) 
  (h1 : r.flour_to_mush = 3)
  (h2 : r.sugar_to_mush = 2/3)
  (h3 : r.milk_to_flour = 1/6)
  (h4 : r.bananas_per_mush = 4)
  (h5 : r.total_bananas = 32) :
  calculate_ingredients r = (24, 16/3, 4) := by
  sorry

#eval calculate_ingredients {
  flour_to_mush := 3,
  sugar_to_mush := 2/3,
  milk_to_flour := 1/6,
  bananas_per_mush := 4,
  total_bananas := 32
}

end correct_ingredients_l3388_338894


namespace pet_farm_hamsters_prove_hamster_count_l3388_338857

/-- Given a pet farm with rabbits and hamsters, prove the number of hamsters. -/
theorem pet_farm_hamsters (rabbit_count : ℕ) (ratio_rabbits : ℕ) (ratio_hamsters : ℕ) : ℕ :=
  let hamster_count := (rabbit_count * ratio_hamsters) / ratio_rabbits
  hamster_count

/-- Prove that there are 25 hamsters given the conditions. -/
theorem prove_hamster_count : pet_farm_hamsters 20 4 5 = 25 := by
  sorry

end pet_farm_hamsters_prove_hamster_count_l3388_338857


namespace robot_cost_calculation_l3388_338839

def number_of_robots : ℕ := 7
def total_tax : ℚ := 7.22
def remaining_change : ℚ := 11.53
def initial_amount : ℚ := 80

theorem robot_cost_calculation (number_of_robots : ℕ) (total_tax : ℚ) (remaining_change : ℚ) (initial_amount : ℚ) :
  let total_spent : ℚ := initial_amount - remaining_change
  let robots_cost : ℚ := total_spent - total_tax
  let cost_per_robot : ℚ := robots_cost / number_of_robots
  cost_per_robot = 8.75 :=
by sorry

end robot_cost_calculation_l3388_338839


namespace vector_difference_l3388_338808

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference --/
theorem vector_difference (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, 3]) (h2 : AC = ![4, 7]) :
  AC - AB = ![2, 4] := by
  sorry

end vector_difference_l3388_338808


namespace craft_item_pricing_problem_l3388_338887

/-- Represents the daily profit function for a craft item store -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

theorem craft_item_pricing_problem 
  (initial_sales : ℕ) 
  (initial_profit : ℝ) 
  (price_reduction_1050 : ℝ) 
  (h1 : initial_sales = 20)
  (h2 : initial_profit = 40)
  (h3 : price_reduction_1050 < 40)
  (h4 : daily_profit initial_sales initial_profit price_reduction_1050 = 1050) :
  price_reduction_1050 = 25 ∧ 
  ∀ (price_reduction : ℝ), price_reduction < 40 → 
    daily_profit initial_sales initial_profit price_reduction ≠ 1600 := by
  sorry


end craft_item_pricing_problem_l3388_338887


namespace exactly_two_good_probability_l3388_338858

def total_screws : ℕ := 10
def defective_screws : ℕ := 3
def drawn_screws : ℕ := 4

def probability_exactly_two_good : ℚ :=
  (Nat.choose (total_screws - defective_screws) 2 * Nat.choose defective_screws 2) /
  Nat.choose total_screws drawn_screws

theorem exactly_two_good_probability :
  probability_exactly_two_good = 3 / 10 := by
  sorry

end exactly_two_good_probability_l3388_338858


namespace trig_identity_l3388_338818

theorem trig_identity (θ : Real) 
  (h : (1 - Real.cos θ) / (4 + Real.sin θ ^ 2) = 1 / 2) : 
  (4 + Real.cos θ ^ 3) * (3 + Real.sin θ ^ 3) = 9 := by
  sorry

end trig_identity_l3388_338818
