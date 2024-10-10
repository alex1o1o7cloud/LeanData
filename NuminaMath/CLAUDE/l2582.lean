import Mathlib

namespace log_equation_solutions_l2582_258224

theorem log_equation_solutions :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) →
  (x^2 - 5*y^2 + 4 = 0) →
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) := by
  sorry

end log_equation_solutions_l2582_258224


namespace runners_alignment_time_l2582_258206

def steinLapTime : ℕ := 6
def roseLapTime : ℕ := 10
def schwartzLapTime : ℕ := 18

theorem runners_alignment_time :
  Nat.lcm steinLapTime (Nat.lcm roseLapTime schwartzLapTime) = 90 := by
  sorry

end runners_alignment_time_l2582_258206


namespace cylinder_cube_surface_equality_l2582_258254

theorem cylinder_cube_surface_equality (r h s K : ℝ) : 
  r = 3 → h = 4 → 
  2 * π * r * h = 6 * s^2 → 
  s^3 = 48 / Real.sqrt K → 
  K = 36 / π^3 := by
sorry

end cylinder_cube_surface_equality_l2582_258254


namespace only_two_special_triples_l2582_258221

/-- A structure representing a triple of positive integers (a, b, c) satisfying certain conditions. -/
structure SpecialTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a ≥ b
  h2 : b ≥ c
  h3 : ∃ x : ℕ, a^2 + 3*b = x^2
  h4 : ∃ y : ℕ, b^2 + 3*c = y^2
  h5 : ∃ z : ℕ, c^2 + 3*a = z^2

/-- The theorem stating that there are only two SpecialTriples. -/
theorem only_two_special_triples :
  {t : SpecialTriple | t.a = 1 ∧ t.b = 1 ∧ t.c = 1} ∪
  {t : SpecialTriple | t.a = 37 ∧ t.b = 25 ∧ t.c = 17} =
  {t : SpecialTriple | True} :=
sorry

end only_two_special_triples_l2582_258221


namespace rhombus_longer_diagonal_l2582_258281

/-- The length of the longer diagonal of a rhombus given its side length and shorter diagonal -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (h1 : side = 65) (h2 : shorter_diagonal = 56) :
  ∃ longer_diagonal : ℝ, longer_diagonal = 2 * Real.sqrt 3441 :=
by sorry

end rhombus_longer_diagonal_l2582_258281


namespace thirteen_ceilings_left_l2582_258258

/-- Represents the number of ceilings left to paint after next week -/
def ceilings_left_after_next_week (stories : ℕ) (rooms_per_floor : ℕ) (ceilings_painted_this_week : ℕ) : ℕ :=
  let total_room_ceilings := stories * rooms_per_floor
  let total_hallway_ceilings := stories
  let total_ceilings := total_room_ceilings + total_hallway_ceilings
  let ceilings_left_after_this_week := total_ceilings - ceilings_painted_this_week
  let ceilings_to_paint_next_week := ceilings_painted_this_week / 4 + stories
  ceilings_left_after_this_week - ceilings_to_paint_next_week

/-- Theorem stating that 13 ceilings will be left to paint after next week -/
theorem thirteen_ceilings_left : ceilings_left_after_next_week 4 7 12 = 13 := by
  sorry

end thirteen_ceilings_left_l2582_258258


namespace sum_geq_sqrt_three_l2582_258244

theorem sum_geq_sqrt_three (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : a + b + c ≥ Real.sqrt 3 := by
  sorry

end sum_geq_sqrt_three_l2582_258244


namespace maya_total_pages_l2582_258256

/-- The number of books Maya read in the first week -/
def first_week_books : ℕ := 5

/-- The number of pages in each book Maya read in the first week -/
def first_week_pages_per_book : ℕ := 300

/-- The number of pages in each book Maya read in the second week -/
def second_week_pages_per_book : ℕ := 350

/-- The number of pages in each book Maya read in the third week -/
def third_week_pages_per_book : ℕ := 400

/-- The total number of pages Maya read over three weeks -/
def total_pages : ℕ :=
  (first_week_books * first_week_pages_per_book) +
  (2 * first_week_books * second_week_pages_per_book) +
  (3 * first_week_books * third_week_pages_per_book)

theorem maya_total_pages : total_pages = 11000 := by
  sorry

end maya_total_pages_l2582_258256


namespace sum_of_digits_l2582_258251

theorem sum_of_digits (a b c d : ℕ) : 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  100 * a + 10 * b + c + 100 * d + 10 * c + a = 1100 →
  a + b + c + d = 21 := by
sorry

end sum_of_digits_l2582_258251


namespace unique_solution_l2582_258240

/-- The functional equation that f must satisfy for all x and y -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that f(x) = 1 - x²/2 is the unique solution -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, functional_equation f ∧ ∀ x : ℝ, f x = 1 - x^2 / 2 := by
  sorry

end unique_solution_l2582_258240


namespace specific_ap_first_term_l2582_258202

/-- An arithmetic progression with given parameters -/
structure ArithmeticProgression where
  n : ℕ             -- number of terms
  d : ℤ             -- common difference
  last_term : ℤ     -- last term

/-- The first term of an arithmetic progression -/
def first_term (ap : ArithmeticProgression) : ℤ :=
  ap.last_term - (ap.n - 1) * ap.d

/-- Theorem stating the first term of the specific arithmetic progression -/
theorem specific_ap_first_term :
  let ap : ArithmeticProgression := ⟨31, 2, 62⟩
  first_term ap = 2 := by sorry

end specific_ap_first_term_l2582_258202


namespace consecutive_even_integers_l2582_258255

theorem consecutive_even_integers (n : ℤ) : 
  (∃ (a b c : ℤ), 
    (a = n - 2 ∧ b = n ∧ c = n + 2) ∧  -- consecutive even integers
    (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) ∧  -- all are even
    (a + c = 128))  -- sum of first and third is 128
  → n = 64 := by
  sorry

end consecutive_even_integers_l2582_258255


namespace jellybean_box_theorem_l2582_258289

/-- The number of jellybeans in a box that is three times larger in each dimension
    compared to a box that holds 200 jellybeans -/
theorem jellybean_box_theorem (ella_jellybeans : ℕ) (scale_factor : ℕ) :
  ella_jellybeans = 200 →
  scale_factor = 3 →
  scale_factor ^ 3 * ella_jellybeans = 5400 :=
by sorry

end jellybean_box_theorem_l2582_258289


namespace mean_temperature_l2582_258238

def temperatures : List ℝ := [79, 81, 83, 85, 84, 86, 88, 87, 85, 84]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 84.2 := by
  sorry

end mean_temperature_l2582_258238


namespace triangle_property_l2582_258275

-- Define a structure for the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.b + t.c = t.a * (Real.cos t.B + Real.cos t.C)) :
  t.A = Real.pi / 2 ∧ 
  Real.sqrt 3 + 2 < 2 * Real.cos (t.B / 2)^2 + 2 * Real.sqrt 3 * Real.cos (t.C / 2)^2 ∧
  2 * Real.cos (t.B / 2)^2 + 2 * Real.sqrt 3 * Real.cos (t.C / 2)^2 ≤ Real.sqrt 3 + 3 := by
  sorry

end triangle_property_l2582_258275


namespace at_least_one_not_less_than_two_l2582_258249

theorem at_least_one_not_less_than_two
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (sum_eq_three : a + b + c = 3) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
by sorry

end at_least_one_not_less_than_two_l2582_258249


namespace min_workers_to_complete_job_l2582_258222

theorem min_workers_to_complete_job
  (total_days : ℕ)
  (days_worked : ℕ)
  (initial_workers : ℕ)
  (job_fraction_completed : ℚ)
  (h1 : total_days = 30)
  (h2 : days_worked = 6)
  (h3 : initial_workers = 8)
  (h4 : job_fraction_completed = 1/3)
  (h5 : days_worked < total_days) :
  ∃ (min_workers : ℕ),
    min_workers ≤ initial_workers ∧
    (min_workers : ℚ) * (total_days - days_worked : ℚ) * job_fraction_completed / days_worked ≥ 1 - job_fraction_completed ∧
    ∀ (w : ℕ), w < min_workers →
      (w : ℚ) * (total_days - days_worked : ℚ) * job_fraction_completed / days_worked < 1 - job_fraction_completed ∧
    min_workers = 4 :=
by sorry

end min_workers_to_complete_job_l2582_258222


namespace parallel_vectors_m_value_l2582_258262

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-2, 1)
  let b : ℝ × ℝ := (m, 3)
  are_parallel a b → m = -6 :=
by sorry

end parallel_vectors_m_value_l2582_258262


namespace power_tower_mod_500_l2582_258280

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end power_tower_mod_500_l2582_258280


namespace polynomial_equality_l2582_258216

theorem polynomial_equality (a b c m n : ℝ) : 
  (∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) →
  a - b + c = 3 := by
sorry

end polynomial_equality_l2582_258216


namespace negation_of_universal_nonnegative_square_l2582_258228

theorem negation_of_universal_nonnegative_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end negation_of_universal_nonnegative_square_l2582_258228


namespace sin_equality_proof_l2582_258208

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → Real.sin (n * π / 180) = Real.sin (721 * π / 180) → n = 1 := by
  sorry

end sin_equality_proof_l2582_258208


namespace victor_trays_l2582_258297

/-- The number of trays Victor can carry per trip -/
def tray_capacity : ℕ := 7

/-- The number of trips Victor made -/
def num_trips : ℕ := 4

/-- The number of trays picked up from the second table -/
def trays_second_table : ℕ := 5

/-- The number of trays picked up from the first table -/
def trays_first_table : ℕ := tray_capacity * num_trips - trays_second_table

theorem victor_trays : trays_first_table = 23 := by
  sorry

end victor_trays_l2582_258297


namespace train_length_l2582_258266

/-- The length of a train given its speed, bridge crossing time, and bridge length -/
theorem train_length (speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 275 →
  speed * crossing_time - bridge_length = 475 :=
by sorry

end train_length_l2582_258266


namespace inequality_proof_l2582_258250

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 2 * c > a + b) : 
  c - Real.sqrt (c^2 - a*b) < a ∧ a < c + Real.sqrt (c^2 - a*b) := by
  sorry

end inequality_proof_l2582_258250


namespace square_area_decrease_l2582_258225

theorem square_area_decrease (initial_side : ℝ) (decrease_percent : ℝ) : 
  initial_side = 9 ∧ decrease_percent = 20 →
  let new_side := initial_side * (1 - decrease_percent / 100)
  let initial_area := initial_side ^ 2
  let new_area := new_side ^ 2
  let area_decrease_percent := (initial_area - new_area) / initial_area * 100
  area_decrease_percent = 36 := by
  sorry

end square_area_decrease_l2582_258225


namespace lowest_number_in_range_l2582_258287

/-- The probability of selecting a number greater than another randomly selected number -/
def probability : ℚ := 4995 / 10000

/-- Theorem stating that given the probability, the lowest number in the range is 999 -/
theorem lowest_number_in_range (x y : ℕ) (h : x ≤ y) :
  (((y - x) * (y - x + 1) : ℚ) / (2 * (y - x + 1)^2)) = probability → x = 999 := by
  sorry

end lowest_number_in_range_l2582_258287


namespace profit_ratio_equals_investment_ratio_l2582_258235

/-- The ratio of profits is equal to the ratio of investments -/
theorem profit_ratio_equals_investment_ratio (p q : ℕ) (h : p = 60000 ∧ q = 90000) : 
  (p : ℚ) / q = 2 / 3 := by sorry

end profit_ratio_equals_investment_ratio_l2582_258235


namespace initial_trees_count_l2582_258279

/-- The number of walnut trees to be removed from the park -/
def trees_removed : ℕ := 4

/-- The number of walnut trees remaining after removal -/
def trees_remaining : ℕ := 2

/-- The initial number of walnut trees in the park -/
def initial_trees : ℕ := trees_removed + trees_remaining

theorem initial_trees_count : initial_trees = 6 := by sorry

end initial_trees_count_l2582_258279


namespace octagon_diagonal_property_l2582_258259

theorem octagon_diagonal_property (x : ℕ) (h : x > 2) :
  (x * (x - 3)) / 2 = x + 2 * (x - 2) → x = 8 := by
  sorry

end octagon_diagonal_property_l2582_258259


namespace injective_function_inequality_l2582_258246

theorem injective_function_inequality (f : ℕ → ℕ) 
  (h_inj : Function.Injective f) 
  (h_ineq : ∀ n : ℕ, f (f n) ≤ (n + f n) / 2) : 
  ∀ n : ℕ, f n = n := by
  sorry

end injective_function_inequality_l2582_258246


namespace expression_evaluation_l2582_258282

theorem expression_evaluation :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end expression_evaluation_l2582_258282


namespace four_liters_possible_l2582_258215

/-- Represents the state of water in two vessels -/
structure WaterState :=
  (small : ℕ)  -- Amount of water in the 3-liter vessel
  (large : ℕ)  -- Amount of water in the 5-liter vessel

/-- Represents a pouring operation between vessels -/
inductive PourOperation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | SmallToLarge
  | LargeToSmall

/-- Applies a pouring operation to a water state -/
def applyOperation (state : WaterState) (op : PourOperation) : WaterState :=
  match op with
  | PourOperation.FillSmall => ⟨3, state.large⟩
  | PourOperation.FillLarge => ⟨state.small, 5⟩
  | PourOperation.EmptySmall => ⟨0, state.large⟩
  | PourOperation.EmptyLarge => ⟨state.small, 0⟩
  | PourOperation.SmallToLarge =>
      let amount := min state.small (5 - state.large)
      ⟨state.small - amount, state.large + amount⟩
  | PourOperation.LargeToSmall =>
      let amount := min state.large (3 - state.small)
      ⟨state.small + amount, state.large - amount⟩

/-- Theorem stating that it's possible to obtain 4 liters in the 5-liter vessel -/
theorem four_liters_possible : ∃ (ops : List PourOperation),
  (ops.foldl applyOperation ⟨0, 0⟩).large = 4 :=
sorry

end four_liters_possible_l2582_258215


namespace min_value_cos_sin_l2582_258290

theorem min_value_cos_sin (θ : Real) (h : θ ∈ Set.Icc (-π/12) (π/12)) :
  ∃ (m : Real), m = (Real.sqrt 3 - 1) / 2 ∧ 
    ∀ x ∈ Set.Icc (-π/12) (π/12), 
      Real.cos (x + π/4) + Real.sin (2*x) ≥ m ∧
      ∃ y ∈ Set.Icc (-π/12) (π/12), Real.cos (y + π/4) + Real.sin (2*y) = m :=
by sorry

end min_value_cos_sin_l2582_258290


namespace problem_1_problem_2_problem_3_problem_4_l2582_258217

-- Problem 1
theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 := by sorry

-- Problem 2
theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 := by sorry

-- Problem 3
theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 := by sorry

-- Problem 4
theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2582_258217


namespace sqrt_meaningful_range_l2582_258207

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 6 + x) ↔ x ≥ -6 := by
  sorry

end sqrt_meaningful_range_l2582_258207


namespace average_speed_calculation_l2582_258292

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  (total_distance / total_time) = 40 := by
  sorry

end average_speed_calculation_l2582_258292


namespace triple_hash_48_l2582_258223

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.75 * N + 2

-- State the theorem
theorem triple_hash_48 : hash (hash (hash 48)) = 24.875 := by
  sorry

end triple_hash_48_l2582_258223


namespace triangle_properties_l2582_258286

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- Define the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : Real.sin (t.A + t.B) = 3/5)
  (h2 : Real.sin (t.A - t.B) = 1/5)
  (h3 : ∃ (AB : Real), AB = 3) :
  (Real.tan t.A = 2 * Real.tan t.B) ∧
  (∃ (height : Real), height = 2 + Real.sqrt 6) := by
  sorry


end triangle_properties_l2582_258286


namespace integer_representation_l2582_258245

theorem integer_representation (n : ℤ) : 
  ∃ (a b c d : ℤ), n = a^2 + b^2 + c^2 + d^2 ∨ n = a^2 + b^2 + c^2 - d^2 ∨
                    n = a^2 + b^2 - c^2 - d^2 ∨ n = a^2 - b^2 - c^2 - d^2 :=
sorry

example : ∃ (a b c : ℤ), 1947 = a^2 - b^2 - c^2 :=
sorry

end integer_representation_l2582_258245


namespace quadratic_equation_solution_l2582_258200

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 3) - (x - 3)
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end quadratic_equation_solution_l2582_258200


namespace a_plus_b_value_l2582_258277

-- Define the functions f and h
def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, h (f a b x) = 5 * x - 8) → a + b = -4/3 := by
  sorry

end a_plus_b_value_l2582_258277


namespace same_solution_implies_m_value_l2582_258209

theorem same_solution_implies_m_value : ∀ m x : ℚ,
  (8 - m = 2 * (x + 1) ∧ 2 * (2 * x - 3) - 1 = 1 - 2 * x) →
  m = 10 / 3 := by
  sorry

end same_solution_implies_m_value_l2582_258209


namespace greatest_multiple_of_four_l2582_258242

theorem greatest_multiple_of_four (y : ℕ) : 
  y > 0 ∧ 
  ∃ k : ℕ, y = 4 * k ∧ 
  y^3 < 4096 →
  y ≤ 12 ∧ 
  ∀ z : ℕ, z > 0 ∧ (∃ m : ℕ, z = 4 * m) ∧ z^3 < 4096 → z ≤ y :=
by sorry

end greatest_multiple_of_four_l2582_258242


namespace sixtieth_pair_is_five_seven_l2582_258234

/-- Represents a pair of integers in the sequence -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def generatePair (n : ℕ) : IntPair :=
  sorry

/-- The main theorem stating that the 60th pair is (5,7) -/
theorem sixtieth_pair_is_five_seven : generatePair 60 = IntPair.mk 5 7 := by
  sorry

end sixtieth_pair_is_five_seven_l2582_258234


namespace carton_weight_is_three_l2582_258298

/-- The weight of one crate of vegetables in kilograms. -/
def crate_weight : ℝ := 4

/-- The number of crates in the load. -/
def num_crates : ℕ := 12

/-- The number of cartons in the load. -/
def num_cartons : ℕ := 16

/-- The total weight of the load in kilograms. -/
def total_weight : ℝ := 96

/-- The weight of one carton of vegetables in kilograms. -/
def carton_weight : ℝ := 3

/-- Theorem stating that the weight of one carton of vegetables is 3 kilograms. -/
theorem carton_weight_is_three :
  crate_weight * num_crates + carton_weight * num_cartons = total_weight :=
by sorry

end carton_weight_is_three_l2582_258298


namespace defective_clock_correct_time_fraction_l2582_258239

/-- Represents a 12-hour digital clock with a defect that displays '7' instead of '1'. -/
structure DefectiveClock where
  /-- The number of hours in the clock cycle -/
  hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The number of hours displayed correctly -/
  correct_hours : Nat
  /-- The number of minutes displayed correctly in each hour -/
  correct_minutes : Nat

/-- The fraction of the day that the defective clock displays the correct time -/
def correct_time_fraction (clock : DefectiveClock) : ℚ :=
  (clock.correct_hours : ℚ) / clock.hours * (clock.correct_minutes : ℚ) / clock.minutes_per_hour

/-- Theorem stating that the fraction of the day the defective clock displays the correct time is 1/2 -/
theorem defective_clock_correct_time_fraction :
  ∃ (clock : DefectiveClock),
    clock.hours = 12 ∧
    clock.minutes_per_hour = 60 ∧
    clock.correct_hours = 8 ∧
    clock.correct_minutes = 45 ∧
    correct_time_fraction clock = 1 / 2 := by
  sorry

end defective_clock_correct_time_fraction_l2582_258239


namespace domain_intersection_l2582_258233

-- Define the domain of y = e^x
def M : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Define the domain of y = ln x
def N : Set ℝ := {y | ∃ x, y = Real.log x}

-- Theorem statement
theorem domain_intersection :
  M ∩ N = {y : ℝ | y > 0} := by sorry

end domain_intersection_l2582_258233


namespace set_difference_equals_singleton_l2582_258276

-- Define the set I
def I : Set ℕ := {x | 1 < x ∧ x < 5}

-- Define the set A
def A : Set ℕ := {2, 3}

-- Theorem statement
theorem set_difference_equals_singleton :
  I \ A = {4} := by sorry

end set_difference_equals_singleton_l2582_258276


namespace quadratic_equation_with_prime_roots_l2582_258263

theorem quadratic_equation_with_prime_roots (a m : ℤ) :
  (∃ x y : ℕ, x ≠ y ∧ Prime x ∧ Prime y ∧ (a * x^2 - m * x + 1996 = 0) ∧ (a * y^2 - m * y + 1996 = 0)) →
  a = 2 := by
  sorry

end quadratic_equation_with_prime_roots_l2582_258263


namespace equations_represent_problem_l2582_258265

/-- Represents the money held by person A -/
def money_A : ℝ := sorry

/-- Represents the money held by person B -/
def money_B : ℝ := sorry

/-- The system of equations representing the problem -/
def problem_equations (x y : ℝ) : Prop :=
  (x + (1/2) * y = 50) ∧ (y + (2/3) * x = 50)

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem equations_represent_problem :
  problem_equations money_A money_B ↔
  ((money_A + (1/2) * money_B = 50) ∧
   (money_B + (2/3) * money_A = 50)) :=
sorry

end equations_represent_problem_l2582_258265


namespace dans_minimum_speed_l2582_258201

/-- Proves that Dan must travel at a speed greater than 48 miles per hour to arrive in city B before Cara. -/
theorem dans_minimum_speed (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) : 
  distance = 120 → 
  cara_speed = 30 → 
  dan_delay = 1.5 → 
  ∀ dan_speed : ℝ, dan_speed > 48 → distance / dan_speed < distance / cara_speed - dan_delay := by
  sorry

#check dans_minimum_speed

end dans_minimum_speed_l2582_258201


namespace parallelogram_area_l2582_258227

/-- The area of a parallelogram with a diagonal of length 30 meters and an altitude of 20 meters to that diagonal is 600 square meters. -/
theorem parallelogram_area (d : ℝ) (h : ℝ) (h1 : d = 30) (h2 : h = 20) :
  d * h = 600 := by sorry

end parallelogram_area_l2582_258227


namespace square_root_sum_equals_six_l2582_258291

theorem square_root_sum_equals_six : 
  Real.sqrt (15 - 6 * Real.sqrt 6) + Real.sqrt (15 + 6 * Real.sqrt 6) = 6 := by
  sorry

end square_root_sum_equals_six_l2582_258291


namespace probability_of_selecting_A_or_B_l2582_258272

-- Define the total number of experts
def total_experts : ℕ := 6

-- Define the number of experts to be selected
def selected_experts : ℕ := 2

-- Define the probability of selecting at least one of A or B
def prob_select_A_or_B : ℚ := 3/5

-- Theorem statement
theorem probability_of_selecting_A_or_B :
  let total_combinations := Nat.choose total_experts selected_experts
  let combinations_without_A_and_B := Nat.choose (total_experts - 2) selected_experts
  1 - (combinations_without_A_and_B : ℚ) / total_combinations = prob_select_A_or_B :=
by sorry

end probability_of_selecting_A_or_B_l2582_258272


namespace doll_count_l2582_258248

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

/-- The number of collector's edition dolls Ivy has -/
def ivy_collector_dolls : ℕ := 20

/-- The number of dolls Casey has -/
def casey_dolls : ℕ := 5 * ivy_collector_dolls

/-- The total number of dolls Dina, Ivy, and Casey have together -/
def total_dolls : ℕ := dina_dolls + ivy_dolls + casey_dolls

theorem doll_count : total_dolls = 190 ∧ 
  2 * ivy_dolls / 3 = ivy_collector_dolls := by
  sorry

end doll_count_l2582_258248


namespace janes_mean_score_l2582_258268

def quiz_scores : List ℝ := [99, 95, 93, 87, 90]
def exam_scores : List ℝ := [88, 92]

def all_scores : List ℝ := quiz_scores ++ exam_scores

theorem janes_mean_score :
  (all_scores.sum / all_scores.length : ℝ) = 644 / 7 := by
  sorry

end janes_mean_score_l2582_258268


namespace arithmetic_calculation_l2582_258264

theorem arithmetic_calculation : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by
  sorry

end arithmetic_calculation_l2582_258264


namespace no_solution_equation_l2582_258231

theorem no_solution_equation : ¬∃ (x : ℝ), x ≠ 2 ∧ x + 5 / (x - 2) = 2 + 5 / (x - 2) := by
  sorry

end no_solution_equation_l2582_258231


namespace cube_sum_product_l2582_258284

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 := by
  sorry

end cube_sum_product_l2582_258284


namespace exam_score_difference_l2582_258212

def math_exam_problem (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ) : Prop :=
  bryan_score = 20 ∧
  jen_score = bryan_score + 10 ∧
  sammy_score < jen_score ∧
  total_points = 35 ∧
  sammy_mistakes = 7 ∧
  sammy_score = total_points - sammy_mistakes ∧
  jen_score - sammy_score = 2

theorem exam_score_difference :
  ∀ (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ),
  math_exam_problem bryan_score jen_score sammy_score total_points sammy_mistakes :=
by
  sorry

#check exam_score_difference

end exam_score_difference_l2582_258212


namespace smallest_three_digit_divisible_by_3_and_6_l2582_258252

theorem smallest_three_digit_divisible_by_3_and_6 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 6 = 0 → n ≥ 102 := by
  sorry

end smallest_three_digit_divisible_by_3_and_6_l2582_258252


namespace min_cost_57_and_227_l2582_258247

/-- Calculates the minimum cost for notebooks given the pricing structure and number of notebooks -/
def min_cost (n : ℕ) : ℚ :=
  let single_price := 0.3
  let dozen_price := 3.0
  let bulk_dozen_price := 2.7
  let dozens := n / 12
  let singles := n % 12
  if dozens > 10 then
    bulk_dozen_price * dozens + single_price * singles
  else if singles = 0 then
    dozen_price * dozens
  else
    min (dozen_price * (dozens + 1)) (dozen_price * dozens + single_price * singles)

theorem min_cost_57_and_227 :
  min_cost 57 = 14.7 ∧ min_cost 227 = 51.3 := by sorry

end min_cost_57_and_227_l2582_258247


namespace multitool_comparison_l2582_258241

-- Define the contents of each multitool
def walmart_tools : ℕ := 2 + 4 + 1 + 1 + 1
def target_knives : ℕ := 4 * 3
def target_tools : ℕ := 3 + target_knives + 2 + 1 + 1 + 2

-- Theorem to prove the difference in tools and the ratio
theorem multitool_comparison :
  (target_tools - walmart_tools = 12) ∧
  (target_tools / walmart_tools = 7 / 3) := by
  sorry

#eval walmart_tools
#eval target_tools

end multitool_comparison_l2582_258241


namespace solve_equation_l2582_258274

theorem solve_equation (x : ℝ) : 3*x + 15 = (1/3) * (8*x + 48) → x = 3 := by
  sorry

end solve_equation_l2582_258274


namespace storm_rainfall_theorem_l2582_258253

/-- Represents the rainfall data for a city over three days -/
structure CityRainfall where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ

/-- Represents the rainfall data for two cities -/
structure StormData where
  cityA : CityRainfall
  cityB : CityRainfall
  X : ℝ  -- Combined rainfall on day 3
  Y : ℝ  -- Total rainfall over three days

/-- Defines the conditions of the storm and proves the results -/
theorem storm_rainfall_theorem (s : StormData) : 
  s.cityA.day1 = 4 ∧ 
  s.cityA.day2 = 5 * s.cityA.day1 ∧
  s.cityB.day2 = 3 * s.cityA.day1 ∧
  s.cityA.day3 = (s.cityA.day1 + s.cityA.day2) / 2 ∧
  s.cityB.day3 = s.cityB.day1 + s.cityB.day2 - 6 ∧
  s.X = s.cityA.day3 + s.cityB.day3 ∧
  s.Y = s.cityA.day1 + s.cityA.day2 + s.cityA.day3 + s.cityB.day1 + s.cityB.day2 + s.cityB.day3 →
  s.cityA.day3 = 12 ∧
  s.cityB.day3 = s.cityB.day1 + 6 ∧
  s.X = 18 + s.cityB.day1 ∧
  s.Y = 54 + 2 * s.cityB.day1 := by
  sorry


end storm_rainfall_theorem_l2582_258253


namespace division_by_fraction_not_always_larger_l2582_258288

theorem division_by_fraction_not_always_larger : ∃ (a b c : ℚ), b ≠ 0 ∧ c ≠ 0 ∧ (a / (b / c)) ≤ a := by
  sorry

end division_by_fraction_not_always_larger_l2582_258288


namespace max_bribe_amount_l2582_258219

/-- Represents the bet amount in coins -/
def betAmount : ℤ := 100

/-- Represents the maximum bribe amount in coins -/
def maxBribe : ℤ := 199

/-- 
Proves that the maximum bribe a person would pay to avoid eviction is 199 coins, 
given a bet where they lose 100 coins if evicted and gain 100 coins if not evicted, 
assuming they act solely in their own financial interest.
-/
theorem max_bribe_amount : 
  ∀ (bribe : ℤ), 
    bribe ≤ maxBribe ∧ 
    bribe > betAmount ∧
    (maxBribe - betAmount ≤ betAmount) ∧
    (∀ (x : ℤ), x > maxBribe → x - betAmount > betAmount) := by
  sorry


end max_bribe_amount_l2582_258219


namespace critical_points_product_bound_l2582_258296

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1/2) * m * x^2 - x

theorem critical_points_product_bound (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∃ (y : ℝ), y ∈ Set.Icc x₁ x₂ ∧ (deriv (f m)) y = 0) →
  (deriv (f m)) x₁ = 0 →
  (deriv (f m)) x₂ = 0 →
  x₁ * x₂ > Real.exp 2 := by
sorry

end critical_points_product_bound_l2582_258296


namespace condition_equivalence_l2582_258283

theorem condition_equivalence (a b : ℝ) (h : |a| > |b|) :
  (a - b > 0) ↔ (a + b > 0) := by
  sorry

end condition_equivalence_l2582_258283


namespace divisible_by_nine_exists_l2582_258273

def is_distinct (digits : List Nat) : Prop :=
  digits.length = digits.toFinset.card

def sum_digits (digits : List Nat) : Nat :=
  digits.sum

theorem divisible_by_nine_exists (kolya_number : List Nat) :
  kolya_number.length = 10 →
  (∀ d ∈ kolya_number, d < 10) →
  is_distinct kolya_number →
  ∃ d : Nat, d < 10 ∧ d ∉ kolya_number ∧
    (sum_digits kolya_number + d) % 9 = 0 :=
by sorry

end divisible_by_nine_exists_l2582_258273


namespace symmetric_points_relation_l2582_258267

/-- 
Given two points P and Q in the 2D plane, where:
- P has coordinates (m+1, 3)
- Q has coordinates (1, n-2)
- P is symmetric to Q with respect to the x-axis

This theorem proves that m-n = 1.
-/
theorem symmetric_points_relation (m n : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    P = (m + 1, 3) ∧ 
    Q = (1, n - 2) ∧ 
    P.1 = Q.1 ∧ 
    P.2 = -Q.2) → 
  m - n = 1 := by
sorry

end symmetric_points_relation_l2582_258267


namespace a_3_equals_negative_10_l2582_258261

def a (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem a_3_equals_negative_10 : a 3 = -10 := by
  sorry

end a_3_equals_negative_10_l2582_258261


namespace cone_volume_approximation_l2582_258237

theorem cone_volume_approximation (r h : ℝ) (π : ℝ) : 
  (1/3) * π * r^2 * h = (2/75) * (2 * π * r)^2 * h → π = 25/8 := by
  sorry

end cone_volume_approximation_l2582_258237


namespace four_Z_three_l2582_258293

def Z (a b : ℤ) : ℤ := a^2 - 3*a*b + b^2

theorem four_Z_three : Z 4 3 = -11 := by
  sorry

end four_Z_three_l2582_258293


namespace simplify_expression_l2582_258214

theorem simplify_expression :
  4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 := by
  sorry

end simplify_expression_l2582_258214


namespace school_distance_l2582_258271

/-- 
Given a person who walks to and from a destination for 5 days, 
with an additional 4km on the last day, and a total distance of 74km,
prove that the one-way distance to the destination is 7km.
-/
theorem school_distance (x : ℝ) 
  (h1 : (4 * 2 * x) + (2 * x + 4) = 74) : x = 7 := by
  sorry

end school_distance_l2582_258271


namespace parabola_vertex_sum_max_l2582_258220

theorem parabola_vertex_sum_max (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x : ℝ) := a * x * (x - 2 * T)
  let N := T - a * T^2
  (parabola 0 = 0) → 
  (parabola (2 * T) = 0) → 
  (parabola (T + 2) = 36) → 
  (∀ (a' T' : ℤ), T' ≠ 0 → 
    let parabola' (x : ℝ) := a' * x * (x - 2 * T')
    let N' := T' - a' * T'^2
    (parabola' 0 = 0) → 
    (parabola' (2 * T') = 0) → 
    (parabola' (T' + 2) = 36) → 
    N ≥ N') → 
  N = 37 :=
sorry

end parabola_vertex_sum_max_l2582_258220


namespace nested_sqrt_fifteen_l2582_258226

theorem nested_sqrt_fifteen (x : ℝ) : x = Real.sqrt (15 + x) → x = (1 + Real.sqrt 61) / 2 := by
  sorry

end nested_sqrt_fifteen_l2582_258226


namespace expression_equality_l2582_258295

theorem expression_equality (x : ℝ) (h : x > 0) : 
  x^x - x^x = 0 ∧ (x - 1)^x = 0 := by
  sorry

end expression_equality_l2582_258295


namespace discount_percentage_l2582_258218

theorem discount_percentage
  (MP : ℝ)
  (CP : ℝ)
  (SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (h2 : (SP - CP) / CP = 0.5454545454545454)
  : (MP - SP) / MP = 0.15 := by
  sorry

end discount_percentage_l2582_258218


namespace bacteria_growth_l2582_258229

/-- Calculates the bacteria population after a given time interval -/
def bacteria_population (initial_population : ℕ) (doubling_time : ℕ) (total_time : ℕ) : ℕ :=
  initial_population * 2 ^ (total_time / doubling_time)

/-- Theorem: Given 20 initial bacteria that double every 3 minutes, 
    the population after 15 minutes is 640 -/
theorem bacteria_growth : bacteria_population 20 3 15 = 640 := by
  sorry

end bacteria_growth_l2582_258229


namespace equation_holds_for_all_x_l2582_258210

theorem equation_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m - 5) * x = 0) → m = 5 := by
  sorry

end equation_holds_for_all_x_l2582_258210


namespace tan_3_75_deg_sum_l2582_258285

theorem tan_3_75_deg_sum (a b c d : ℕ+) 
  (h1 : Real.tan (3.75 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - d)
  (h2 : a ≥ b) (h3 : b ≥ c) (h4 : c ≥ d) :
  a + b + c + d = 13 := by
  sorry

end tan_3_75_deg_sum_l2582_258285


namespace catering_company_comparison_l2582_258257

/-- Represents the cost function for a catering company -/
structure CateringCompany where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (company : CateringCompany) (people : ℕ) : ℕ :=
  company.basicFee + company.perPersonFee * people

/-- The problem statement -/
theorem catering_company_comparison :
  let company1 : CateringCompany := ⟨120, 18⟩
  let company2 : CateringCompany := ⟨250, 15⟩
  ∀ n : ℕ, n < 44 → totalCost company1 n ≤ totalCost company2 n ∧
  totalCost company2 44 < totalCost company1 44 :=
by sorry

end catering_company_comparison_l2582_258257


namespace concert_revenue_is_930_l2582_258203

/-- Calculates the total revenue for a concert given the number of tickets sold and their prices. -/
def concert_revenue (student_tickets : ℕ) (non_student_tickets : ℕ) (student_price : ℕ) (non_student_price : ℕ) : ℕ :=
  student_tickets * student_price + non_student_tickets * non_student_price

/-- Proves that the total revenue for the concert is $930 given the specified conditions. -/
theorem concert_revenue_is_930 :
  let total_tickets : ℕ := 150
  let student_price : ℕ := 5
  let non_student_price : ℕ := 8
  let student_tickets : ℕ := 90
  let non_student_tickets : ℕ := 60
  concert_revenue student_tickets non_student_tickets student_price non_student_price = 930 :=
by
  sorry

#eval concert_revenue 90 60 5 8

end concert_revenue_is_930_l2582_258203


namespace latia_hourly_wage_l2582_258270

/-- The cost of the TV in dollars -/
def tv_cost : ℝ := 1700

/-- The number of hours Latia works per week -/
def weekly_hours : ℝ := 30

/-- The additional hours Latia needs to work to afford the TV -/
def additional_hours : ℝ := 50

/-- The number of weeks in a month -/
def weeks_per_month : ℝ := 4

/-- Latia's hourly wage in dollars -/
def hourly_wage : ℝ := 10

theorem latia_hourly_wage :
  tv_cost = (weekly_hours * weeks_per_month + additional_hours) * hourly_wage :=
by sorry

end latia_hourly_wage_l2582_258270


namespace tom_payment_l2582_258299

/-- The total amount Tom paid for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 965 for his purchase -/
theorem tom_payment : total_amount 8 70 9 45 = 965 := by
  sorry

end tom_payment_l2582_258299


namespace coin_collection_dimes_l2582_258260

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

theorem coin_collection_dimes :
  ∀ (p n d q h : ℕ),
    p ≥ 1 → n ≥ 1 → d ≥ 1 → q ≥ 1 → h ≥ 1 →
    p + n + d + q + h = 12 →
    p * penny + n * nickel + d * dime + q * quarter + h * half_dollar = 163 →
    d = 5 := by
  sorry

end coin_collection_dimes_l2582_258260


namespace annual_increase_y_l2582_258232

/-- The annual increase in price of commodity Y -/
def y : ℝ := sorry

/-- The price of commodity X in a given year -/
def price_x (year : ℕ) : ℝ :=
  4.20 + 0.30 * (year - 2001)

/-- The price of commodity Y in a given year -/
def price_y (year : ℕ) : ℝ :=
  4.40 + y * (year - 2001)

theorem annual_increase_y : y = 0.20 :=
  have h1 : price_x 2010 = price_y 2010 + 0.70 := by sorry
  sorry

end annual_increase_y_l2582_258232


namespace area_of_DEFGHT_l2582_258204

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a square -/
structure Square :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Function to calculate the area of a shape formed by DEFGHT -/
def areaOfDEFGHT (ABC : Triangle) (ABDE : Square) (CAFG : Square) (BCHT : Triangle) : ℝ :=
  sorry

/-- Theorem stating the area of shape DEFGHT -/
theorem area_of_DEFGHT :
  ∀ (ABC : Triangle) (ABDE : Square) (CAFG : Square) (BCHT : Triangle),
  (ABC.A.x - ABC.B.x)^2 + (ABC.A.y - ABC.B.y)^2 = 4 ∧  -- Side length of ABC is 2
  (ABC.B.x - ABC.C.x)^2 + (ABC.B.y - ABC.C.y)^2 = 4 ∧
  (ABC.C.x - ABC.A.x)^2 + (ABC.C.y - ABC.A.y)^2 = 4 ∧
  (ABDE.A = ABC.A ∧ ABDE.B = ABC.B) ∧  -- ABDE is a square outside ABC
  (CAFG.C = ABC.A ∧ CAFG.A = ABC.C) ∧  -- CAFG is a square outside ABC
  (BCHT.B = ABC.B ∧ BCHT.C = ABC.C) ∧  -- BCHT is an equilateral triangle outside ABC
  (BCHT.A.x - BCHT.B.x)^2 + (BCHT.A.y - BCHT.B.y)^2 = 4 ∧  -- BCHT is equilateral with side length 2
  (BCHT.B.x - BCHT.C.x)^2 + (BCHT.B.y - BCHT.C.y)^2 = 4 ∧
  (BCHT.C.x - BCHT.A.x)^2 + (BCHT.C.y - BCHT.A.y)^2 = 4 →
  areaOfDEFGHT ABC ABDE CAFG BCHT = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end area_of_DEFGHT_l2582_258204


namespace parabola_circle_triangle_l2582_258269

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by x^2 = 2py -/
def Parabola (p : ℝ) : Set Point :=
  {pt : Point | pt.x^2 = 2 * p * pt.y}

/-- Check if three points form an equilateral triangle -/
def isEquilateralTriangle (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = 
  (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (b.x - c.x)^2 + (b.y - c.y)^2 = 
  (c.x - a.x)^2 + (c.y - a.y)^2

/-- The origin point -/
def O : Point := ⟨0, 0⟩

/-- The given point M -/
def M : Point := ⟨0, 9⟩

theorem parabola_circle_triangle (p : ℝ) 
  (h_p_pos : p > 0)
  (A : Point)
  (h_A_on_parabola : A ∈ Parabola p)
  (B : Point)
  (h_B_on_parabola : B ∈ Parabola p)
  (h_circle : (A.x - M.x)^2 + (A.y - M.y)^2 = A.x^2 + A.y^2)
  (h_equilateral : isEquilateralTriangle A B O) :
  p = 3/4 := by
  sorry


end parabola_circle_triangle_l2582_258269


namespace friends_drawing_cards_l2582_258236

theorem friends_drawing_cards (n : ℕ) (h : n = 3) :
  let total_outcomes := n.factorial
  let favorable_outcomes := (n - 1).factorial
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by sorry

end friends_drawing_cards_l2582_258236


namespace ellipse_m_range_l2582_258243

/-- An ellipse is represented by the equation x²/(25 - m) + y²/(m + 9) = 1 with foci on the y-axis -/
def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b > a ∧ 
  (∀ (x y : ℝ), x^2 / (25 - m) + y^2 / (m + 9) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

/-- The range of m for the given ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse_with_y_foci m ↔ 8 < m ∧ m < 25 := by sorry

end ellipse_m_range_l2582_258243


namespace white_balls_count_l2582_258294

theorem white_balls_count (black_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) : 
  black_balls = 6 →
  prob_white = 45454545454545453 / 100000000000000000 →
  (white_balls : ℚ) / ((black_balls : ℚ) + (white_balls : ℚ)) = prob_white →
  white_balls = 5 := by
sorry

end white_balls_count_l2582_258294


namespace reporters_covering_local_politics_l2582_258230

theorem reporters_covering_local_politics
  (percent_not_covering_local : Real)
  (percent_not_covering_politics : Real)
  (h1 : percent_not_covering_local = 0.3)
  (h2 : percent_not_covering_politics = 0.6) :
  (1 - percent_not_covering_politics) * (1 - percent_not_covering_local) = 0.28 := by
  sorry

end reporters_covering_local_politics_l2582_258230


namespace intersection_of_A_and_B_l2582_258211

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2582_258211


namespace ball_hitting_ground_time_l2582_258278

theorem ball_hitting_ground_time :
  let height (t : ℝ) := -16 * t^2 + 16 * t + 50
  ∃ t : ℝ, t > 0 ∧ height t = 0 ∧ t = (2 + 3 * Real.sqrt 6) / 4 := by
  sorry

end ball_hitting_ground_time_l2582_258278


namespace triangle_vector_ratio_l2582_258205

/-- Given a triangle ABC with point E, prove that if AE = 3/4 * AB + 1/4 * AC, 
    then BE = 1/3 * EC -/
theorem triangle_vector_ratio (A B C E : ℝ × ℝ) : 
  (E - A) = 3/4 * (B - A) + 1/4 * (C - A) → 
  (E - B) = 1/3 * (C - E) := by
  sorry

end triangle_vector_ratio_l2582_258205


namespace cricket_team_size_l2582_258213

/-- Represents a cricket team with given age properties -/
structure CricketTeam where
  n : ℕ  -- number of team members
  captain_age : ℕ
  wicket_keeper_age : ℕ
  team_avg_age : ℝ
  remaining_avg_age : ℝ

/-- The cricket team satisfies the given conditions -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.captain_age = 26 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.team_avg_age = 23 ∧
  team.remaining_avg_age = team.team_avg_age - 1 ∧
  (team.n : ℝ) * team.team_avg_age = 
    (team.n - 2 : ℝ) * team.remaining_avg_age + team.captain_age + team.wicket_keeper_age

theorem cricket_team_size (team : CricketTeam) :
  valid_cricket_team team → team.n = 11 := by
  sorry

end cricket_team_size_l2582_258213
