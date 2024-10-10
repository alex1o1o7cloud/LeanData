import Mathlib

namespace negation_of_existence_l4029_402984

theorem negation_of_existence (n : ℝ) :
  (¬ ∃ a : ℝ, a ≥ -1 ∧ Real.log (Real.exp n + 1) > 1/2) ↔
  (∀ a : ℝ, a ≥ -1 → Real.log (Real.exp n + 1) ≤ 1/2) :=
by sorry

end negation_of_existence_l4029_402984


namespace largest_power_dividing_product_l4029_402933

-- Define pow function
def pow (n : ℕ) : ℕ := sorry

-- Define the product of pow(n) from 2 to 5300
def product : ℕ := sorry

-- State the theorem
theorem largest_power_dividing_product : 
  (∃ m : ℕ, (2010 ^ m : ℕ) ∣ product ∧ 
   ∀ k : ℕ, k > m → ¬((2010 ^ k : ℕ) ∣ product)) ∧ 
  (∃ m : ℕ, m = 77 ∧ (2010 ^ m : ℕ) ∣ product ∧ 
   ∀ k : ℕ, k > m → ¬((2010 ^ k : ℕ) ∣ product)) := by
  sorry

end largest_power_dividing_product_l4029_402933


namespace root_minus_one_implies_k_eq_neg_two_l4029_402973

theorem root_minus_one_implies_k_eq_neg_two (k : ℝ) :
  ((-1 : ℝ)^2 - k*(-1) + 1 = 0) → k = -2 :=
by sorry

end root_minus_one_implies_k_eq_neg_two_l4029_402973


namespace currency_notes_count_l4029_402945

/-- Given a total amount of currency notes and specific conditions, 
    prove the total number of notes. -/
theorem currency_notes_count 
  (total_amount : ℕ) 
  (denomination_70 : ℕ) 
  (denomination_50 : ℕ) 
  (amount_in_50 : ℕ) 
  (h1 : total_amount = 5000)
  (h2 : denomination_70 = 70)
  (h3 : denomination_50 = 50)
  (h4 : amount_in_50 = 100)
  (h5 : ∃ (x y : ℕ), denomination_70 * x + denomination_50 * y = total_amount ∧ 
                     denomination_50 * (amount_in_50 / denomination_50) = amount_in_50) :
  ∃ (x y : ℕ), denomination_70 * x + denomination_50 * y = total_amount ∧ x + y = 72 := by
  sorry

end currency_notes_count_l4029_402945


namespace jane_sarah_age_sum_l4029_402946

theorem jane_sarah_age_sum : 
  ∀ (jane sarah : ℝ),
  jane = sarah + 5 →
  jane + 9 = 3 * (sarah - 3) →
  jane + sarah = 28 :=
by
  sorry

end jane_sarah_age_sum_l4029_402946


namespace circumscribed_sphere_surface_area_is_32pi_l4029_402950

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  PA : ℝ
  AB : ℝ
  BC : ℝ
  angleABC : ℝ

/-- The surface area of the circumscribed sphere of a triangular pyramid -/
def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the circumscribed sphere for the given pyramid -/
theorem circumscribed_sphere_surface_area_is_32pi :
  let pyramid : TriangularPyramid := {
    PA := 4,
    AB := 2,
    BC := 2,
    angleABC := 2 * Real.pi / 3  -- 120° in radians
  }
  circumscribedSphereSurfaceArea pyramid = 32 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_is_32pi_l4029_402950


namespace mono_increasing_minus_decreasing_mono_decreasing_minus_increasing_l4029_402935

-- Define monotonically increasing function
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Theorem for proposition ②
theorem mono_increasing_minus_decreasing
  (f g : ℝ → ℝ) (hf : MonoIncreasing f) (hg : MonoDecreasing g) :
  MonoIncreasing (fun x ↦ f x - g x) :=
sorry

-- Theorem for proposition ③
theorem mono_decreasing_minus_increasing
  (f g : ℝ → ℝ) (hf : MonoDecreasing f) (hg : MonoIncreasing g) :
  MonoDecreasing (fun x ↦ f x - g x) :=
sorry

end mono_increasing_minus_decreasing_mono_decreasing_minus_increasing_l4029_402935


namespace border_mass_of_28_coin_triangle_l4029_402957

/-- Represents a triangular arrangement of coins -/
structure CoinTriangle where
  total_coins : ℕ
  border_coins : ℕ
  trio_mass : ℝ

/-- The mass of all border coins in a CoinTriangle -/
def border_mass (ct : CoinTriangle) : ℝ := sorry

/-- Theorem stating the mass of border coins in the specific arrangement -/
theorem border_mass_of_28_coin_triangle (ct : CoinTriangle) 
  (h1 : ct.total_coins = 28)
  (h2 : ct.border_coins = 18)
  (h3 : ct.trio_mass = 10) :
  border_mass ct = 60 := by sorry

end border_mass_of_28_coin_triangle_l4029_402957


namespace turtles_jumped_off_l4029_402982

/-- The fraction of turtles that jumped off the log --/
def fraction_jumped_off (initial : ℕ) (remaining : ℕ) : ℚ :=
  let additional := 3 * initial - 2
  let total := initial + additional
  (total - remaining) / total

/-- Theorem stating that the fraction of turtles that jumped off is 1/2 --/
theorem turtles_jumped_off :
  fraction_jumped_off 9 17 = 1 / 2 := by
  sorry

end turtles_jumped_off_l4029_402982


namespace album_photos_l4029_402919

theorem album_photos (n : ℕ) 
  (h1 : ∀ (album : ℕ), album > 0 → ∃ (page : ℕ), page > 0 ∧ page ≤ n)
  (h2 : ∀ (page : ℕ), page > 0 → page ≤ n → ∃ (photos : Fin 4), True)
  (h3 : ∃ (album : ℕ), album > 0 ∧ 81 ∈ Set.range (λ i => 4*(n*(album-1) + 5) - 3 + i) ∧ (∀ j, j ∈ Set.range (λ i => 4*(n*(album-1) + 5) - 3 + i) → j ≤ 4*n*album))
  (h4 : ∃ (album : ℕ), album > 0 ∧ 171 ∈ Set.range (λ i => 4*(n*(album-1) + 3) - 3 + i) ∧ (∀ j, j ∈ Set.range (λ i => 4*(n*(album-1) + 3) - 3 + i) → j ≤ 4*n*album))
  : n = 8 ∧ 4*n = 32 := by
  sorry

end album_photos_l4029_402919


namespace square_sum_equality_l4029_402972

theorem square_sum_equality (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end square_sum_equality_l4029_402972


namespace max_power_under_500_l4029_402980

theorem max_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 1) (hab : a^b < 500) :
  (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → a^b ≥ c^d) →
  a = 22 ∧ b = 2 ∧ a + b = 24 :=
sorry

end max_power_under_500_l4029_402980


namespace hotel_light_bulbs_l4029_402979

theorem hotel_light_bulbs 
  (I F : ℕ) -- I: number of incandescent bulbs, F: number of fluorescent bulbs
  (h_positive : I > 0 ∧ F > 0) -- ensure positive numbers of bulbs
  (h_incandescent_on : (3 : ℝ) / 10 * I = (1 : ℝ) / 7 * (7 : ℝ) / 10 * (I + F)) -- 30% of incandescent on, which is 1/7 of all on bulbs
  (h_total_on : (7 : ℝ) / 10 * (I + F) = (3 : ℝ) / 10 * I + x * F) -- 70% of all bulbs are on
  (x : ℝ) -- x is the fraction of fluorescent bulbs that are on
  : x = (9 : ℝ) / 10 := by
sorry

end hotel_light_bulbs_l4029_402979


namespace cos_theta_plus_pi_fourth_l4029_402965

theorem cos_theta_plus_pi_fourth (θ : ℝ) (h : Real.sin (θ - π/4) = 1/5) : 
  Real.cos (θ + π/4) = -1/5 := by
  sorry

end cos_theta_plus_pi_fourth_l4029_402965


namespace consecutive_page_numbers_sum_l4029_402971

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end consecutive_page_numbers_sum_l4029_402971


namespace equation_solution_l4029_402999

theorem equation_solution : 
  let S : Set ℝ := {x | 3 * x * (x - 2) = 2 * (x - 2)}
  S = {2/3, 2} := by sorry

end equation_solution_l4029_402999


namespace colored_copies_count_l4029_402997

/-- Represents the number of copies and their costs --/
structure CopyData where
  totalCopies : ℕ
  regularHoursCopies : ℕ
  coloredRegularCost : ℚ
  coloredAfterHoursCost : ℚ
  whiteCopyCost : ℚ
  totalBill : ℚ

/-- Theorem stating that given the conditions, the number of colored copies is 300 --/
theorem colored_copies_count (data : CopyData)
  (h1 : data.totalCopies = 400)
  (h2 : data.regularHoursCopies = 180)
  (h3 : data.coloredRegularCost = 10/100)
  (h4 : data.coloredAfterHoursCost = 8/100)
  (h5 : data.whiteCopyCost = 5/100)
  (h6 : data.totalBill = 45/2)
  : ∃ (coloredCopies : ℕ), coloredCopies = 300 ∧ 
    (coloredCopies : ℚ) * data.coloredRegularCost * (data.regularHoursCopies : ℚ) / data.totalCopies +
    (coloredCopies : ℚ) * data.coloredAfterHoursCost * (data.totalCopies - data.regularHoursCopies : ℚ) / data.totalCopies +
    (data.totalCopies - coloredCopies : ℚ) * data.whiteCopyCost = data.totalBill :=
sorry

end colored_copies_count_l4029_402997


namespace not_prime_for_all_positive_n_l4029_402978

def f (n : ℕ+) : ℤ := (n : ℤ)^3 - 9*(n : ℤ)^2 + 23*(n : ℤ) - 17

theorem not_prime_for_all_positive_n : ∀ n : ℕ+, ¬(Nat.Prime (Int.natAbs (f n))) := by
  sorry

end not_prime_for_all_positive_n_l4029_402978


namespace children_born_in_current_marriage_l4029_402958

/-- Represents the number of children in a blended family scenario -/
structure BlendedFamily where
  x : ℕ  -- children from father's previous marriage
  y : ℕ  -- children from mother's previous marriage
  z : ℕ  -- children born in current marriage
  total_children : x + y + z = 12
  father_bio_children : x + z = 9
  mother_bio_children : y + z = 9

/-- Theorem stating that in this blended family scenario, 6 children were born in the current marriage -/
theorem children_born_in_current_marriage (family : BlendedFamily) : family.z = 6 := by
  sorry

#check children_born_in_current_marriage

end children_born_in_current_marriage_l4029_402958


namespace cubic_equation_solution_l4029_402942

theorem cubic_equation_solution (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 :=
by sorry

end cubic_equation_solution_l4029_402942


namespace even_function_implies_a_zero_l4029_402927

noncomputable def f (a x : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, x > (1/2) ∨ x < -(1/2) → f a x = f a (-x)) → a = 0 :=
by sorry

end even_function_implies_a_zero_l4029_402927


namespace union_of_A_and_B_l4029_402966

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end union_of_A_and_B_l4029_402966


namespace nut_weight_l4029_402956

/-- A proof that determines the weight of a nut attached to a scale -/
theorem nut_weight (wL wS : ℝ) (h1 : wL + 20 = 300) (h2 : wS + 20 = 200) (h3 : wL + wS + 20 = 480) : 20 = 20 := by
  sorry

end nut_weight_l4029_402956


namespace cube_root_of_negative_eight_l4029_402904

theorem cube_root_of_negative_eight :
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by sorry

end cube_root_of_negative_eight_l4029_402904


namespace factorial_7_base_9_trailing_zeros_l4029_402928

-- Define 7!
def factorial_7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the base conversion function (simplified)
noncomputable def to_base_9 (n : ℕ) : List ℕ :=
  sorry  -- Actual implementation would go here

-- Define a function to count trailing zeros
def count_trailing_zeros (digits : List ℕ) : ℕ :=
  sorry  -- Actual implementation would go here

-- Theorem statement
theorem factorial_7_base_9_trailing_zeros :
  count_trailing_zeros (to_base_9 factorial_7) = 1 :=
sorry

end factorial_7_base_9_trailing_zeros_l4029_402928


namespace sum_of_squares_and_square_of_sum_l4029_402964

theorem sum_of_squares_and_square_of_sum : (3 + 5)^2 + (3^2 + 5^2) = 98 := by
  sorry

end sum_of_squares_and_square_of_sum_l4029_402964


namespace tangent_segment_length_l4029_402907

theorem tangent_segment_length (r : ℝ) (a b : ℝ) : 
  r = 15 ∧ a = 6 ∧ b = 3 →
  ∃ x : ℝ, x = 12 ∧
    r^2 = x^2 + ((x + r - a - b) / 2)^2 ∧
    x + r = a + b + x + r - a - b :=
by sorry

end tangent_segment_length_l4029_402907


namespace log_35_28_in_terms_of_a_and_b_l4029_402934

theorem log_35_28_in_terms_of_a_and_b (a b : ℝ) 
  (h1 : Real.log 7 / Real.log 14 = a) 
  (h2 : Real.log 5 / Real.log 14 = b) : 
  Real.log 28 / Real.log 35 = (2 - a) / (a + b) := by
  sorry

end log_35_28_in_terms_of_a_and_b_l4029_402934


namespace log_inequality_l4029_402990

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| > |Real.log b|) : a * b < 1 := by
  sorry

end log_inequality_l4029_402990


namespace a_properties_l4029_402959

/-- Sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1  -- a_1 = 1
  | 1 => 6  -- a_2 = 6
  | (n+2) => ((n+3) * (a (n+1) - 1)) / (n+2)

/-- Theorem stating the properties of sequence a_n -/
theorem a_properties :
  (∀ n : ℕ, a n = 2 * n^2 - n) ∧
  (∃ p q : ℚ, p ≠ 0 ∧ q ≠ 0 ∧
    (∃ d : ℚ, ∀ n : ℕ, a (n+1) / (p * (n+1) + q) - a n / (p * n + q) = d) ↔
    p + 2*q = 0) := by sorry

end a_properties_l4029_402959


namespace evaluate_expression_l4029_402909

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l4029_402909


namespace equation_solution_l4029_402931

theorem equation_solution : ∃ x : ℚ, (2 / 5 - 1 / 3 : ℚ) = 1 / x ∧ x = 15 := by
  sorry

end equation_solution_l4029_402931


namespace housing_boom_proof_l4029_402948

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_proof : houses_built = 574 := by
  sorry

end housing_boom_proof_l4029_402948


namespace blonde_girls_count_l4029_402976

/-- Represents the choir composition -/
structure Choir :=
  (initial_total : ℕ)
  (added_blonde : ℕ)
  (black_haired : ℕ)

/-- Calculates the initial number of blonde-haired girls in the choir -/
def initial_blonde (c : Choir) : ℕ :=
  c.initial_total - c.black_haired

/-- Theorem stating the initial number of blonde-haired girls in the specific choir -/
theorem blonde_girls_count (c : Choir) 
  (h1 : c.initial_total = 80)
  (h2 : c.added_blonde = 10)
  (h3 : c.black_haired = 50) :
  initial_blonde c = 30 := by
  sorry

end blonde_girls_count_l4029_402976


namespace people_on_boats_l4029_402903

/-- Given 5 boats in a lake, each with 3 people, prove that the total number of people on boats is 15. -/
theorem people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) 
  (h1 : num_boats = 5) 
  (h2 : people_per_boat = 3) : 
  num_boats * people_per_boat = 15 := by
  sorry

end people_on_boats_l4029_402903


namespace roots_product_theorem_l4029_402994

theorem roots_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 16/3 := by
sorry

end roots_product_theorem_l4029_402994


namespace product_of_solutions_l4029_402992

theorem product_of_solutions (x : ℝ) : 
  (∃ α β : ℝ, (α * β = -10) ∧ (10 = -α^2 - 4*α) ∧ (10 = -β^2 - 4*β)) := by
  sorry

end product_of_solutions_l4029_402992


namespace can_display_rows_l4029_402996

/-- Represents a display of cans arranged in rows -/
structure CanDisplay where
  topRowCans : ℕ
  rowIncrement : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can display -/
def numberOfRows (display : CanDisplay) : ℕ :=
  sorry

/-- Theorem: A display with 3 cans in the top row, 2 more cans in each subsequent row, 
    and 225 total cans has 15 rows -/
theorem can_display_rows (display : CanDisplay) 
  (h1 : display.topRowCans = 3)
  (h2 : display.rowIncrement = 2)
  (h3 : display.totalCans = 225) : 
  numberOfRows display = 15 := by
  sorry

end can_display_rows_l4029_402996


namespace find_M_l4029_402955

theorem find_M : ∃ M : ℕ, (9.5 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) ∧ M = 39 := by
  sorry

end find_M_l4029_402955


namespace josh_recording_time_l4029_402926

/-- A device that records temperature data at regular intervals. -/
structure TemperatureRecorder where
  interval : ℕ  -- Recording interval in seconds
  instances : ℕ  -- Number of recorded instances

/-- Calculates the total recording time in hours for a TemperatureRecorder. -/
def totalRecordingTime (recorder : TemperatureRecorder) : ℚ :=
  (recorder.interval * recorder.instances : ℚ) / 3600

/-- Theorem: Josh's device recorded data for 1 hour. -/
theorem josh_recording_time :
  let device : TemperatureRecorder := { interval := 5, instances := 720 }
  totalRecordingTime device = 1 := by sorry

end josh_recording_time_l4029_402926


namespace outfits_count_l4029_402918

/-- The number of different outfits that can be made from a given number of shirts, ties, and shoes. -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (shoes : ℕ) : ℕ :=
  shirts * ties * shoes

/-- Theorem stating that the number of outfits is 192 given 8 shirts, 6 ties, and 4 pairs of shoes. -/
theorem outfits_count : number_of_outfits 8 6 4 = 192 := by
  sorry

end outfits_count_l4029_402918


namespace no_high_grades_l4029_402930

/-- Represents the test scenario with given conditions -/
structure TestScenario where
  n : ℕ  -- number of students excluding Peter
  k : ℕ  -- number of problems solved by each student except Peter
  total_problems_solved : ℕ  -- total number of problems solved by all students

/-- The conditions of the test scenario -/
def valid_scenario (s : TestScenario) : Prop :=
  s.total_problems_solved = 25 ∧
  s.n * s.k + (s.k + 1) = s.total_problems_solved ∧
  s.k ≤ 5

/-- The theorem stating that no student received a grade of 4 or 5 -/
theorem no_high_grades (s : TestScenario) (h : valid_scenario s) : 
  s.k < 4 ∧ s.k + 1 < 5 := by
  sorry

#check no_high_grades

end no_high_grades_l4029_402930


namespace scooter_gain_percent_l4029_402937

theorem scooter_gain_percent (initial_cost repair1 repair2 repair3 selling_price : ℚ) : 
  initial_cost = 800 →
  repair1 = 150 →
  repair2 = 75 →
  repair3 = 225 →
  selling_price = 1600 →
  let total_cost := initial_cost + repair1 + repair2 + repair3
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 28 := by
sorry

end scooter_gain_percent_l4029_402937


namespace willies_bananas_unchanged_l4029_402932

/-- Willie's banana count remains unchanged regardless of Charles' banana count changes -/
theorem willies_bananas_unchanged (willie_initial : ℕ) (charles_initial charles_lost : ℕ) :
  willie_initial = 48 → willie_initial = willie_initial :=
by
  sorry

end willies_bananas_unchanged_l4029_402932


namespace square_of_difference_101_minus_2_l4029_402912

theorem square_of_difference_101_minus_2 :
  (101 - 2)^2 = 9801 := by
  sorry

end square_of_difference_101_minus_2_l4029_402912


namespace stratified_sampling_ratio_l4029_402968

theorem stratified_sampling_ratio (total : ℕ) (first_year : ℕ) (second_year : ℕ) (selected_first : ℕ) :
  total = first_year + second_year →
  first_year = 30 →
  second_year = 40 →
  selected_first = 6 →
  (selected_first * second_year) / first_year = 8 :=
by sorry

end stratified_sampling_ratio_l4029_402968


namespace probability_two_non_defective_pens_l4029_402916

/-- The probability of selecting two non-defective pens from a box of pens -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 4) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  (total_pens - defective_pens - 1) / (total_pens - 1) = 14 / 33 := by
  sorry

#check probability_two_non_defective_pens

end probability_two_non_defective_pens_l4029_402916


namespace tetrahedron_volume_ratio_l4029_402924

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedronVolume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 12

/-- Theorem: The volume ratio of two regular tetrahedrons with edge lengths a and 2a is 1:8 -/
theorem tetrahedron_volume_ratio (a : ℝ) (h : a > 0) :
  tetrahedronVolume (2 * a) / tetrahedronVolume a = 8 := by
  sorry

end tetrahedron_volume_ratio_l4029_402924


namespace windy_driving_time_l4029_402920

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  non_windy_speed : ℝ  -- Speed in non-windy conditions (miles per hour)
  windy_speed : ℝ      -- Speed in windy conditions (miles per hour)
  total_distance : ℝ   -- Total distance covered (miles)
  total_time : ℝ       -- Total time spent driving (minutes)

/-- Calculates the time spent driving in windy conditions -/
def time_in_windy_conditions (scenario : DrivingScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the time spent in windy conditions is 20 minutes -/
theorem windy_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.non_windy_speed = 40)
  (h2 : scenario.windy_speed = 25)
  (h3 : scenario.total_distance = 25)
  (h4 : scenario.total_time = 45) :
  time_in_windy_conditions scenario = 20 := by
  sorry

end windy_driving_time_l4029_402920


namespace three_inequality_propositions_l4029_402905

theorem three_inequality_propositions (a b c d : ℝ) :
  (∃ (f g h : Prop),
    (f = (a * b > 0)) ∧
    (g = (c / a > d / b)) ∧
    (h = (b * c > a * d)) ∧
    ((f ∧ g → h) ∧ (f ∧ h → g) ∧ (g ∧ h → f)) ∧
    (∀ (p q r : Prop),
      ((p = f ∨ p = g ∨ p = h) ∧
       (q = f ∨ q = g ∨ q = h) ∧
       (r = f ∨ r = g ∨ r = h) ∧
       (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r) ∧
       (p ∧ q → r)) →
      ((p = f ∧ q = g ∧ r = h) ∨
       (p = f ∧ q = h ∧ r = g) ∨
       (p = g ∧ q = h ∧ r = f)))) :=
by sorry

end three_inequality_propositions_l4029_402905


namespace smallest_positive_integer_l4029_402970

theorem smallest_positive_integer : ∀ n : ℕ, n > 0 → n ≥ 1 := by
  sorry

end smallest_positive_integer_l4029_402970


namespace triangle_properties_l4029_402941

/-- Given a, b, and c are side lengths of a triangle, prove the following properties --/
theorem triangle_properties (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) :
  (a + b - c > 0) ∧ 
  (a - b + c > 0) ∧ 
  (a - b - c < 0) ∧
  (|a + b - c| - |a - b + c| + |a - b - c| = -a + 3*b - c) := by
  sorry

end triangle_properties_l4029_402941


namespace circle_center_coordinate_product_l4029_402922

/-- Given a circle with equation x^2 + y^2 = 6x + 10y - 14, 
    the product of its center coordinates is 15 -/
theorem circle_center_coordinate_product : 
  ∀ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 = 6*x + 10*y - 14 → (x - h)^2 + (y - k)^2 = 20) → 
  h * k = 15 := by
  sorry

end circle_center_coordinate_product_l4029_402922


namespace probability_of_event_B_l4029_402908

theorem probability_of_event_B 
  (P_A : ℝ) 
  (P_A_and_B : ℝ) 
  (P_A_or_B : ℝ) 
  (h1 : P_A = 0.4)
  (h2 : P_A_and_B = 0.25)
  (h3 : P_A_or_B = 0.6) :
  P_A + (P_A_or_B - P_A + P_A_and_B) - P_A_and_B = 0.45 := by
  sorry

end probability_of_event_B_l4029_402908


namespace inconsistent_farm_animals_l4029_402911

theorem inconsistent_farm_animals :
  ∀ (x y z g : ℕ),
  x = 2 * y →
  y = 310 →
  z = 180 →
  x + y + z + g = 900 →
  g < 0 :=
by
  sorry

end inconsistent_farm_animals_l4029_402911


namespace prime_divisor_of_fermat_number_l4029_402962

theorem prime_divisor_of_fermat_number (p k : ℕ) : 
  Prime p → p ∣ (2^(2^k) + 1) → (2^(k+1) ∣ (p - 1)) := by
  sorry

end prime_divisor_of_fermat_number_l4029_402962


namespace profit_percentage_calculation_l4029_402915

def cost_price : ℝ := 180
def selling_price : ℝ := 207

theorem profit_percentage_calculation :
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 15 := by
sorry

end profit_percentage_calculation_l4029_402915


namespace common_first_digit_is_two_l4029_402967

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n
  else first_digit (n / 10)

def three_digit_powers_of_2 : Set ℕ :=
  {n | ∃ m : ℕ, n = 2^m ∧ is_three_digit n}

def three_digit_powers_of_3 : Set ℕ :=
  {n | ∃ m : ℕ, n = 3^m ∧ is_three_digit n}

theorem common_first_digit_is_two :
  ∃! d : ℕ, (∃ n ∈ three_digit_powers_of_2, first_digit n = d) ∧
            (∃ m ∈ three_digit_powers_of_3, first_digit m = d) ∧
            d = 2 :=
sorry

end common_first_digit_is_two_l4029_402967


namespace highlighter_profit_l4029_402914

/-- Calculates the profit from selling highlighter pens under specific conditions --/
theorem highlighter_profit : 
  let total_boxes : ℕ := 12
  let pens_per_box : ℕ := 30
  let cost_per_box : ℕ := 10
  let rearranged_boxes : ℕ := 5
  let pens_per_package : ℕ := 6
  let price_per_package : ℕ := 3
  let pens_per_group : ℕ := 3
  let price_per_group : ℕ := 2

  let total_cost : ℕ := total_boxes * cost_per_box
  let total_pens : ℕ := total_boxes * pens_per_box
  let packages : ℕ := rearranged_boxes * (pens_per_box / pens_per_package)
  let revenue_packages : ℕ := packages * price_per_package
  let remaining_pens : ℕ := total_pens - (rearranged_boxes * pens_per_box)
  let groups : ℕ := remaining_pens / pens_per_group
  let revenue_groups : ℕ := groups * price_per_group
  let total_revenue : ℕ := revenue_packages + revenue_groups
  let profit : ℕ := total_revenue - total_cost

  profit = 115 := by sorry

end highlighter_profit_l4029_402914


namespace boxes_problem_l4029_402938

theorem boxes_problem (stan jules joseph john : ℕ) : 
  stan = 100 →
  joseph = stan / 5 →
  jules = joseph + 5 →
  john = jules + jules / 5 →
  john = 30 := by
sorry

end boxes_problem_l4029_402938


namespace school_sections_l4029_402917

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := by
sorry

end school_sections_l4029_402917


namespace quadratic_inequality_range_l4029_402969

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + k^2 - 3 > 0) ↔ (k > 2 ∨ k < -2) := by
sorry

end quadratic_inequality_range_l4029_402969


namespace melon_count_l4029_402949

/-- Given the number of watermelons and apples, calculate the number of melons -/
theorem melon_count (watermelons apples : ℕ) (h1 : watermelons = 3) (h2 : apples = 7) :
  2 * (watermelons + apples) = 20 := by
  sorry

end melon_count_l4029_402949


namespace base5_1204_eq_179_l4029_402913

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 5^3 + d₂ * 5^2 + d₁ * 5^1 + d₀ * 5^0

/-- Proves that 1204₍₅₎ is equal to 179 in decimal --/
theorem base5_1204_eq_179 : base5ToDecimal 1 2 0 4 = 179 := by
  sorry

end base5_1204_eq_179_l4029_402913


namespace parallel_iff_m_eq_neg_two_l4029_402960

-- Define the lines as functions of x and y
def line1 (m : ℝ) (x y : ℝ) : Prop := 2*x + m*y - 2*m + 4 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m*x + 2*y - m + 2 = 0

-- Define what it means for two lines to be parallel
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y ↔ ∃ (k : ℝ), line2 m (x + k) (y + k)

-- State the theorem
theorem parallel_iff_m_eq_neg_two :
  ∀ m : ℝ, parallel m ↔ m = -2 := by sorry

end parallel_iff_m_eq_neg_two_l4029_402960


namespace cube_root_of_negative_eight_l4029_402951

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l4029_402951


namespace robot_return_distance_l4029_402981

/-- A robot's walk pattern -/
structure RobotWalk where
  step_distance : ℝ
  turn_angle : ℝ

/-- The total angle turned by the robot -/
def total_angle (w : RobotWalk) (n : ℕ) : ℝ := n * w.turn_angle

/-- The distance walked by the robot -/
def total_distance (w : RobotWalk) (n : ℕ) : ℝ := n * w.step_distance

/-- Theorem: A robot walking 1m and turning left 45° each time will return to its starting point after 8 steps -/
theorem robot_return_distance (w : RobotWalk) (h1 : w.step_distance = 1) (h2 : w.turn_angle = 45) :
  ∃ n : ℕ, total_angle w n = 360 ∧ total_distance w n = 8 := by
  sorry

end robot_return_distance_l4029_402981


namespace total_wheels_is_25_l4029_402921

/-- The number of wheels in Zoe's garage --/
def total_wheels : ℕ :=
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 4
  let num_unicycles : ℕ := 7
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  let wheels_per_unicycle : ℕ := 1
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle

theorem total_wheels_is_25 : total_wheels = 25 := by
  sorry

end total_wheels_is_25_l4029_402921


namespace product_fourth_minus_seven_l4029_402936

theorem product_fourth_minus_seven (a b c d : ℕ) (h₁ : a = 5) (h₂ : b = 9) (h₃ : c = 4) (h₄ : d = 7) :
  (a * b * c : ℚ) / 4 - d = 38 := by
  sorry

end product_fourth_minus_seven_l4029_402936


namespace women_fair_hair_percentage_is_twenty_percent_l4029_402985

/-- Represents the percentage of fair-haired employees who are women -/
def fair_haired_women_ratio : ℝ := 0.4

/-- Represents the percentage of employees who have fair hair -/
def fair_haired_ratio : ℝ := 0.5

/-- Calculates the percentage of employees who are women with fair hair -/
def women_fair_hair_percentage : ℝ := fair_haired_women_ratio * fair_haired_ratio

theorem women_fair_hair_percentage_is_twenty_percent :
  women_fair_hair_percentage = 0.2 := by sorry

end women_fair_hair_percentage_is_twenty_percent_l4029_402985


namespace ten_ways_to_distribute_albums_l4029_402963

/-- Represents the number of ways to distribute albums to friends -/
def distribute_albums (photo_albums : ℕ) (stamp_albums : ℕ) (friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 10 ways to distribute 4 albums to 4 friends -/
theorem ten_ways_to_distribute_albums :
  distribute_albums 2 3 4 = 10 := by
  sorry

end ten_ways_to_distribute_albums_l4029_402963


namespace square_sum_and_product_l4029_402991

theorem square_sum_and_product (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 := by
sorry

end square_sum_and_product_l4029_402991


namespace max_value_constraint_l4029_402944

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 4) :
  (10 * x + 3 * y + 15 * z)^2 ≤ 3220 / 36 :=
by sorry

end max_value_constraint_l4029_402944


namespace sphere_cube_surface_area_comparison_l4029_402977

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

noncomputable def cube_volume (a : ℝ) : ℝ := a^3
noncomputable def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

theorem sphere_cube_surface_area_comparison 
  (r a : ℝ) 
  (h_positive : r > 0 ∧ a > 0) 
  (h_equal_volume : sphere_volume r = cube_volume a) : 
  cube_surface_area a > sphere_surface_area r :=
by
  sorry

#check sphere_cube_surface_area_comparison

end sphere_cube_surface_area_comparison_l4029_402977


namespace ice_cream_flavors_l4029_402974

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 5 scoops from 3 basic flavors -/
theorem ice_cream_flavors : distribute 5 3 = 21 := by sorry

end ice_cream_flavors_l4029_402974


namespace max_roses_is_317_l4029_402952

/-- Represents the price of roses in cents to avoid floating-point issues -/
def individual_price : ℕ := 530
def dozen_price : ℕ := 3600
def two_dozen_price : ℕ := 5000
def budget : ℕ := 68000

/-- Calculates the maximum number of roses that can be purchased with the given budget -/
def max_roses : ℕ :=
  let two_dozen_sets := budget / two_dozen_price
  let remaining_budget := budget - two_dozen_sets * two_dozen_price
  let individual_roses := remaining_budget / individual_price
  two_dozen_sets * 24 + individual_roses

theorem max_roses_is_317 : max_roses = 317 := by sorry

end max_roses_is_317_l4029_402952


namespace correct_mean_problem_l4029_402947

def correct_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - incorrect_value + correct_value) / n

theorem correct_mean_problem :
  correct_mean 20 36 40 25 = 35.25 := by
  sorry

end correct_mean_problem_l4029_402947


namespace sum_of_squares_inequality_l4029_402900

theorem sum_of_squares_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 := by
  sorry

end sum_of_squares_inequality_l4029_402900


namespace no_single_liar_l4029_402925

-- Define the propositions
variable (J : Prop) -- Jean is lying
variable (P : Prop) -- Pierre is lying

-- Jean's statement: "When I am not lying, you are not lying either"
axiom jean_statement : ¬J → ¬P

-- Pierre's statement: "When I am lying, you are lying too"
axiom pierre_statement : P → J

-- Theorem: It's impossible for one to be lying and the other not
theorem no_single_liar : ¬(J ∧ ¬P) ∧ ¬(¬J ∧ P) := by
  sorry


end no_single_liar_l4029_402925


namespace max_squares_covered_proof_l4029_402923

/-- The side length of a checkerboard square in inches -/
def checkerboard_square_side : ℝ := 1.25

/-- The side length of the square card in inches -/
def card_side : ℝ := 1.75

/-- The maximum number of checkerboard squares that can be covered by the card -/
def max_squares_covered : ℕ := 9

/-- Theorem stating the maximum number of squares that can be covered by the card -/
theorem max_squares_covered_proof :
  ∀ (card_placement : ℝ × ℝ → Bool),
  (∃ (covered_squares : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ covered_squares →
      ∃ (x y : ℝ), 0 ≤ x ∧ x < card_side ∧ 0 ≤ y ∧ y < card_side ∧
        card_placement (x + i * checkerboard_square_side, y + j * checkerboard_square_side)) ∧
    covered_squares.card ≤ max_squares_covered) ∧
  (∃ (optimal_placement : ℝ × ℝ → Bool) (optimal_covered_squares : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ optimal_covered_squares →
      ∃ (x y : ℝ), 0 ≤ x ∧ x < card_side ∧ 0 ≤ y ∧ y < card_side ∧
        optimal_placement (x + i * checkerboard_square_side, y + j * checkerboard_square_side)) ∧
    optimal_covered_squares.card = max_squares_covered) :=
by sorry

end max_squares_covered_proof_l4029_402923


namespace greatest_root_of_g_l4029_402983

def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/7) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end greatest_root_of_g_l4029_402983


namespace proposition_q_undetermined_l4029_402998

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) ∧ ¬(q ∧ ¬q) := by
sorry

end proposition_q_undetermined_l4029_402998


namespace cutting_process_result_l4029_402954

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with side length -/
structure Square where
  side : ℕ

/-- Cuts the largest possible square from a rectangle and returns the remaining rectangle -/
def cutSquare (r : Rectangle) : Square × Rectangle :=
  if r.width ≤ r.height then
    ({ side := r.width }, { width := r.width, height := r.height - r.width })
  else
    ({ side := r.height }, { width := r.width - r.height, height := r.height })

/-- Applies the cutting process to a rectangle and returns the list of resulting squares -/
def cutProcess (r : Rectangle) : List Square :=
  sorry

/-- Theorem stating the result of applying the cutting process to a 14 × 36 rectangle -/
theorem cutting_process_result :
  let initial_rectangle : Rectangle := { width := 14, height := 36 }
  let result := cutProcess initial_rectangle
  (result.filter (λ s => s.side = 14)).length = 2 ∧
  (result.filter (λ s => s.side = 8)).length = 1 ∧
  (result.filter (λ s => s.side = 6)).length = 1 ∧
  (result.filter (λ s => s.side = 2)).length = 3 :=
sorry

end cutting_process_result_l4029_402954


namespace horner_method_v3_l4029_402986

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 - x^3 + 3x^2 + 7 -/
def f (x : ℚ) : ℚ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_v3 :
  let coeffs := [2, -1, 3, 0, 7]
  let x := 3
  horner coeffs x = 54 ∧ f x = horner coeffs x := by sorry

#check horner_method_v3

end horner_method_v3_l4029_402986


namespace max_three_digit_sum_not_factor_l4029_402953

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_not_factor_of_product (n : ℕ) : Prop :=
  ¬(2 * Nat.factorial (n - 1)) % (n + 1) = 0

theorem max_three_digit_sum_not_factor :
  ∃ (n : ℕ), is_three_digit n ∧ sum_not_factor_of_product n ∧
  ∀ (m : ℕ), is_three_digit m → sum_not_factor_of_product m → m ≤ n :=
by sorry

end max_three_digit_sum_not_factor_l4029_402953


namespace least_subtraction_for_divisibility_l4029_402989

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 11 ∧ (427398 - x) % 12 = 0 ∧ ∀ y : ℕ, y < x → (427398 - y) % 12 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l4029_402989


namespace five_digit_twice_divisible_by_11_l4029_402901

theorem five_digit_twice_divisible_by_11 (a : ℕ) (h : 10000 ≤ a ∧ a < 100000) :
  ∃ k : ℕ, 100001 * a = 11 * k := by
  sorry

end five_digit_twice_divisible_by_11_l4029_402901


namespace jacob_tank_fill_time_l4029_402987

/-- Represents the number of days needed to fill a water tank -/
def days_to_fill_tank (tank_capacity_liters : ℕ) (rain_collection_ml : ℕ) (river_collection_ml : ℕ) : ℕ :=
  (tank_capacity_liters * 1000) / (rain_collection_ml + river_collection_ml)

/-- Theorem stating that it takes 20 days to fill Jacob's water tank -/
theorem jacob_tank_fill_time :
  days_to_fill_tank 50 800 1700 = 20 := by
  sorry

end jacob_tank_fill_time_l4029_402987


namespace expression_evaluation_l4029_402939

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹) = (a * b)⁻¹ :=
by sorry

end expression_evaluation_l4029_402939


namespace gcf_of_180_252_315_l4029_402902

theorem gcf_of_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end gcf_of_180_252_315_l4029_402902


namespace integer_pair_condition_l4029_402906

theorem integer_pair_condition (m n : ℕ+) :
  (∃ k : ℤ, (3 * n.val ^ 2 : ℚ) / m.val = k) ∧
  (∃ l : ℕ, (n.val ^ 2 + m.val : ℕ) = l ^ 2) →
  ∃ a : ℕ+, n = a ∧ m = 3 * a ^ 2 := by
sorry

end integer_pair_condition_l4029_402906


namespace xy_sum_squared_l4029_402988

theorem xy_sum_squared (x y : ℝ) (h1 : x * y = -3) (h2 : x + y = -4) :
  x^2 + 3*x*y + y^2 = 13 := by
sorry

end xy_sum_squared_l4029_402988


namespace adam_book_purchase_l4029_402995

/-- The number of books Adam bought on his shopping trip -/
def books_bought : ℕ := sorry

/-- The number of books Adam had before shopping -/
def initial_books : ℕ := 56

/-- The number of shelves in Adam's bookcase -/
def num_shelves : ℕ := 4

/-- The average number of books per shelf in Adam's bookcase -/
def avg_books_per_shelf : ℕ := 20

/-- The number of books left over after filling the bookcase -/
def leftover_books : ℕ := 2

/-- The theorem stating how many books Adam bought -/
theorem adam_book_purchase :
  books_bought = 
    num_shelves * avg_books_per_shelf + leftover_books - initial_books :=
by sorry

end adam_book_purchase_l4029_402995


namespace complex_magnitude_l4029_402993

theorem complex_magnitude (z : ℂ) : Complex.abs (z - (1 + 2*I)) = 0 → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l4029_402993


namespace log_equation_solution_l4029_402961

theorem log_equation_solution :
  ∃! x : ℝ, Real.log (3 * x + 4) = 1 :=
by
  use 2
  sorry

end log_equation_solution_l4029_402961


namespace bhanu_petrol_expense_l4029_402929

theorem bhanu_petrol_expense (income : ℝ) (petrol_percent house_rent_percent : ℝ) 
  (house_rent : ℝ) : 
  petrol_percent = 0.3 →
  house_rent_percent = 0.1 →
  house_rent = 70 →
  house_rent_percent * (income - petrol_percent * income) = house_rent →
  petrol_percent * income = 300 :=
by sorry

end bhanu_petrol_expense_l4029_402929


namespace cookies_distribution_l4029_402943

/-- Represents the number of cookies the oldest son gets after school -/
def oldest_son_cookies : ℕ := 4

/-- Represents the number of cookies the youngest son gets after school -/
def youngest_son_cookies : ℕ := 2

/-- Represents the total number of cookies in a box -/
def cookies_in_box : ℕ := 54

/-- Represents the number of days the box lasts -/
def days_box_lasts : ℕ := 9

theorem cookies_distribution :
  oldest_son_cookies * days_box_lasts + youngest_son_cookies * days_box_lasts = cookies_in_box :=
by sorry

end cookies_distribution_l4029_402943


namespace first_half_speed_l4029_402975

/-- Proves that given a 60-mile trip where the average speed on the second half is 16 mph faster
    than the first half, and the average speed for the entire trip is 30 mph,
    the average speed during the first half is 24 mph. -/
theorem first_half_speed (total_distance : ℝ) (speed_increase : ℝ) (total_avg_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_increase = 16)
  (h3 : total_avg_speed = 30) :
  ∃ (first_half_speed : ℝ),
    first_half_speed > 0 ∧
    (total_distance / 2) / first_half_speed + (total_distance / 2) / (first_half_speed + speed_increase) = total_distance / total_avg_speed ∧
    first_half_speed = 24 :=
by sorry

end first_half_speed_l4029_402975


namespace xy_bounds_l4029_402910

/-- Given a system of equations x + y = a and x^2 + y^2 = -a^2 + 2,
    prove that the product xy is bounded by -1 ≤ xy ≤ 1/3 -/
theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end xy_bounds_l4029_402910


namespace triangle_inequality_l4029_402940

/-- A triangle with heights and an internal point -/
structure Triangle where
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  l_a : ℝ
  l_b : ℝ
  l_c : ℝ
  h_a_pos : h_a > 0
  h_b_pos : h_b > 0
  h_c_pos : h_c > 0
  l_a_pos : l_a > 0
  l_b_pos : l_b > 0
  l_c_pos : l_c > 0

/-- The inequality holds for any triangle -/
theorem triangle_inequality (t : Triangle) :
  t.h_a / t.l_a + t.h_b / t.l_b + t.h_c / t.l_c ≥ 9 := by
  sorry

end triangle_inequality_l4029_402940
