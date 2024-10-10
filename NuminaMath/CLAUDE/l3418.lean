import Mathlib

namespace sport_water_amount_l3418_341825

/-- Represents the ratios in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard drink formulation -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport drink formulation -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- The amount of corn syrup in the sport formulation -/
def sport_corn_syrup : ℚ := 3

theorem sport_water_amount :
  let water_amount := sport_corn_syrup * sport_ratio.water / sport_ratio.corn_syrup
  water_amount = 45 := by sorry

end sport_water_amount_l3418_341825


namespace sqrt_difference_comparison_l3418_341806

theorem sqrt_difference_comparison (m : ℝ) (hm : m > 1) :
  Real.sqrt m - Real.sqrt (m - 1) > Real.sqrt (m + 1) - Real.sqrt m := by
  sorry

end sqrt_difference_comparison_l3418_341806


namespace max_M_is_five_l3418_341837

/-- Definition of I_k -/
def I (k : ℕ) : ℕ := 10^(k+2) + 25

/-- Definition of M(k) -/
def M (k : ℕ) : ℕ := (I k).factors.count 2

/-- Theorem: The maximum value of M(k) for k > 0 is 5 -/
theorem max_M_is_five : ∃ (k : ℕ), k > 0 ∧ M k = 5 ∧ ∀ (j : ℕ), j > 0 → M j ≤ 5 := by
  sorry

end max_M_is_five_l3418_341837


namespace seat_3_9_description_l3418_341820

/-- Represents a seat in a movie theater -/
structure TheaterSeat where
  row : ℕ
  seat : ℕ

/-- Interprets a pair of natural numbers as a theater seat -/
def interpretSeat (p : ℕ × ℕ) : TheaterSeat :=
  { row := p.1, seat := p.2 }

/-- Describes a theater seat as a string -/
def describeSeat (s : TheaterSeat) : String :=
  s.row.repr ++ "th row, " ++ s.seat.repr ++ "th seat"

theorem seat_3_9_description :
  describeSeat (interpretSeat (3, 9)) = "3rd row, 9th seat" :=
sorry

end seat_3_9_description_l3418_341820


namespace tanya_work_days_l3418_341804

/-- Given Sakshi can do a piece of work in 20 days and Tanya is 25% more efficient than Sakshi,
    prove that Tanya will take 16 days to do the same piece of work. -/
theorem tanya_work_days (sakshi_days : ℕ) (tanya_efficiency : ℚ) :
  sakshi_days = 20 →
  tanya_efficiency = 125 / 100 →
  (sakshi_days : ℚ) / tanya_efficiency = 16 := by
  sorry

end tanya_work_days_l3418_341804


namespace rationalize_denominator_l3418_341815

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℚ), 
    F > 0 ∧ 
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = -1 ∧ B = -3 ∧ C = 1 ∧ D = 2/3 ∧ E = 165 ∧ F = 17 := by
  sorry

end rationalize_denominator_l3418_341815


namespace inverse_composition_l3418_341818

-- Define the function f
def f : ℕ → ℕ
| 2 => 8
| 3 => 15
| 4 => 24
| 5 => 35
| 6 => 48
| _ => 0  -- For other inputs, we'll define it as 0

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 8 => 2
| 15 => 3
| 24 => 4
| 35 => 5
| 48 => 6
| _ => 0  -- For other inputs, we'll define it as 0

-- State the theorem
theorem inverse_composition :
  f_inv (f_inv 48 * f_inv 8 - f_inv 24) = 2 :=
by sorry

end inverse_composition_l3418_341818


namespace quadratic_solution_symmetry_l3418_341817

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_solution_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c
  f (-5) = f 1 → f 2 = 0 → ∃ n : ℝ, f 3 = n ∧ (∀ x : ℝ, f x = n ↔ x = 3 ∨ x = -7) :=
by sorry

end quadratic_solution_symmetry_l3418_341817


namespace daisy_milk_leftover_l3418_341872

/-- Calculates the amount of milk left over given the total production, percentage consumed by kids, and percentage of remainder used for cooking. -/
def milk_left_over (total_milk : ℝ) (kids_consumption_percent : ℝ) (cooking_percent : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption_percent)
  remaining_after_kids * (1 - cooking_percent)

/-- Theorem stating that given 16 cups of milk per day, with 75% consumed by kids and 50% of the remainder used for cooking, 2 cups of milk are left over. -/
theorem daisy_milk_leftover :
  milk_left_over 16 0.75 0.5 = 2 := by
  sorry

#eval milk_left_over 16 0.75 0.5

end daisy_milk_leftover_l3418_341872


namespace polynomial_transformation_l3418_341878

theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 :=
by sorry

end polynomial_transformation_l3418_341878


namespace domain_all_reals_l3418_341827

theorem domain_all_reals (k : ℝ) :
  (∀ x : ℝ, (-7 * x^2 - 4 * x + k ≠ 0)) ↔ k < -4/7 := by sorry

end domain_all_reals_l3418_341827


namespace popsicle_stick_count_l3418_341816

/-- The number of popsicle sticks Steve has -/
def steve_sticks : ℕ := 12

/-- The number of popsicle sticks Sid has -/
def sid_sticks : ℕ := 2 * steve_sticks

/-- The number of popsicle sticks Sam has -/
def sam_sticks : ℕ := 3 * sid_sticks

/-- The total number of popsicle sticks -/
def total_sticks : ℕ := steve_sticks + sid_sticks + sam_sticks

theorem popsicle_stick_count : total_sticks = 108 := by
  sorry

end popsicle_stick_count_l3418_341816


namespace polynomial_value_theorem_l3418_341836

theorem polynomial_value_theorem (f : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (|f 1| = 12 ∧ |f 2| = 12 ∧ |f 3| = 12 ∧ |f 5| = 12 ∧ |f 6| = 12 ∧ |f 7| = 12) →
  |f 0| = 72 := by
sorry

end polynomial_value_theorem_l3418_341836


namespace multiplication_problem_solution_l3418_341871

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  equation : (100 * A.val + 10 * B.val + A.val) * (10 * B.val + C.val) = 
              1000 * B.val + 100 * C.val + 10 * B.val + C.val

theorem multiplication_problem_solution (p : MultiplicationProblem) : 
  p.A.val + p.C.val = 5 := by sorry

end multiplication_problem_solution_l3418_341871


namespace equation_equivalent_to_lines_l3418_341875

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Theorem statement
theorem equation_equivalent_to_lines :
  ∀ x y : ℝ, original_equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end equation_equivalent_to_lines_l3418_341875


namespace exactly_one_two_defective_mutually_exclusive_at_least_one_defective_all_genuine_mutually_exclusive_mutually_exclusive_pairs_l3418_341834

/-- Represents the number of genuine items in the box -/
def genuine_items : ℕ := 4

/-- Represents the number of defective items in the box -/
def defective_items : ℕ := 3

/-- Represents the number of items randomly selected -/
def selected_items : ℕ := 2

/-- Represents the event "Exactly one defective item" -/
def exactly_one_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "Exactly two defective items" -/
def exactly_two_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "At least one defective item" -/
def at_least_one_defective : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Represents the event "All are genuine" -/
def all_genuine : Set (Fin genuine_items × Fin defective_items) := sorry

/-- Theorem stating that "Exactly one defective item" and "Exactly two defective items" are mutually exclusive -/
theorem exactly_one_two_defective_mutually_exclusive :
  exactly_one_defective ∩ exactly_two_defective = ∅ := sorry

/-- Theorem stating that "At least one defective item" and "All are genuine" are mutually exclusive -/
theorem at_least_one_defective_all_genuine_mutually_exclusive :
  at_least_one_defective ∩ all_genuine = ∅ := sorry

/-- Main theorem proving that only the specified pairs of events are mutually exclusive -/
theorem mutually_exclusive_pairs :
  (exactly_one_defective ∩ exactly_two_defective = ∅) ∧
  (at_least_one_defective ∩ all_genuine = ∅) ∧
  (exactly_one_defective ∩ at_least_one_defective ≠ ∅) ∧
  (exactly_two_defective ∩ at_least_one_defective ≠ ∅) ∧
  (exactly_one_defective ∩ all_genuine ≠ ∅) ∧
  (exactly_two_defective ∩ all_genuine ≠ ∅) := sorry

end exactly_one_two_defective_mutually_exclusive_at_least_one_defective_all_genuine_mutually_exclusive_mutually_exclusive_pairs_l3418_341834


namespace negation_of_set_implication_l3418_341801

theorem negation_of_set_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B ≠ A → A ∩ B ≠ B) :=
sorry

end negation_of_set_implication_l3418_341801


namespace arithmetic_sequence_sum_l3418_341803

/-- Given an arithmetic sequence {aₙ}, prove that S₂₀₁₀ = 1005 under the given conditions -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * (a 1 + a n) / 2) → -- Definition of Sₙ
  (∃ O A B C : ℝ × ℝ, 
    B - O = a 1005 • (A - O) + a 1006 • (C - O) ∧ -- Vector equation
    ∃ t : ℝ, B = t • A + (1 - t) • C ∧ -- Collinearity condition
    t ≠ 0 ∧ t ≠ 1) → -- Line doesn't pass through O
  S 2010 = 1005 := by
  sorry

end arithmetic_sequence_sum_l3418_341803


namespace alpha_values_l3418_341824

/-- Given a function f where f(α) = 4, prove that α is either -4 or 2 -/
theorem alpha_values (f : ℝ → ℝ) (α : ℝ) (h : f α = 4) : α = -4 ∨ α = 2 := by
  sorry

end alpha_values_l3418_341824


namespace square_distance_l3418_341838

theorem square_distance (small_perimeter : ℝ) (large_area : ℝ) :
  small_perimeter = 8 →
  large_area = 36 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let leg1 := large_side
  let leg2 := large_side - 2 * small_side
  Real.sqrt (leg1^2 + leg2^2) = 2 * Real.sqrt 10 := by
  sorry

end square_distance_l3418_341838


namespace problem_solution_l3418_341882

theorem problem_solution (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 := by
  sorry

end problem_solution_l3418_341882


namespace hyperbola_focal_length_l3418_341865

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and an asymptote y = 2x/3,
    the focal length is 2√13. -/
theorem hyperbola_focal_length (m : ℝ) :
  (∃ (x y : ℝ), x^2/9 - y^2/m = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), y = 2*x/3) →         -- Asymptote equation
  2 * Real.sqrt 13 = 2 * Real.sqrt ((9:ℝ) + m) := by
  sorry

end hyperbola_focal_length_l3418_341865


namespace k_gt_one_sufficient_k_gt_one_not_necessary_k_gt_one_sufficient_not_necessary_l3418_341879

/-- The equation of a possible hyperbola -/
def hyperbola_equation (k x y : ℝ) : Prop :=
  x^2 / (k - 1) - y^2 / (k + 1) = 1

/-- Condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (k - 1) * (k + 1) > 0

/-- k > 1 is sufficient for the equation to represent a hyperbola -/
theorem k_gt_one_sufficient (k : ℝ) (h : k > 1) : is_hyperbola k := by sorry

/-- k > 1 is not necessary for the equation to represent a hyperbola -/
theorem k_gt_one_not_necessary : ∃ k : ℝ, is_hyperbola k ∧ ¬(k > 1) := by sorry

/-- k > 1 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem k_gt_one_sufficient_not_necessary :
  (∀ k : ℝ, k > 1 → is_hyperbola k) ∧ (∃ k : ℝ, is_hyperbola k ∧ ¬(k > 1)) := by sorry

end k_gt_one_sufficient_k_gt_one_not_necessary_k_gt_one_sufficient_not_necessary_l3418_341879


namespace complex_modulus_problem_l3418_341898

theorem complex_modulus_problem (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 2*a*i)*i = 1 - b*i) : 
  Complex.abs (a + b*i) = Real.sqrt 5 / 2 := by
  sorry

end complex_modulus_problem_l3418_341898


namespace f_t_ratio_is_power_of_two_l3418_341811

/-- Define f_t(n) as the number of odd C_k^t for 1 ≤ k ≤ n -/
def f_t (t n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem f_t_ratio_is_power_of_two (t : ℕ) (h : ℕ) :
  t > 0 → ∃ r : ℕ, ∀ n : ℕ, n = 2^h → (f_t t n : ℚ) / n = 1 / (2^r) := by
  sorry

end f_t_ratio_is_power_of_two_l3418_341811


namespace adults_average_age_is_22_l3418_341852

/-- Represents the programming bootcamp group -/
structure BootcampGroup where
  totalMembers : ℕ
  averageAge : ℕ
  girlsCount : ℕ
  boysCount : ℕ
  adultsCount : ℕ
  girlsAverageAge : ℕ
  boysAverageAge : ℕ

/-- Calculates the average age of adults in the bootcamp group -/
def adultsAverageAge (group : BootcampGroup) : ℕ :=
  ((group.totalMembers * group.averageAge) - 
   (group.girlsCount * group.girlsAverageAge) - 
   (group.boysCount * group.boysAverageAge)) / group.adultsCount

/-- Theorem stating that the average age of adults is 22 years -/
theorem adults_average_age_is_22 (group : BootcampGroup) 
  (h1 : group.totalMembers = 50)
  (h2 : group.averageAge = 20)
  (h3 : group.girlsCount = 25)
  (h4 : group.boysCount = 20)
  (h5 : group.adultsCount = 5)
  (h6 : group.girlsAverageAge = 18)
  (h7 : group.boysAverageAge = 22) :
  adultsAverageAge group = 22 := by
  sorry


end adults_average_age_is_22_l3418_341852


namespace triangle_side_length_l3418_341859

theorem triangle_side_length (a b : ℝ) (A B : ℝ) : 
  a = 4 →
  A = π / 3 →  -- 60° in radians
  B = π / 4 →  -- 45° in radians
  b = (4 * Real.sqrt 6) / 3 :=
by sorry

end triangle_side_length_l3418_341859


namespace complex_magnitude_fourth_power_l3418_341866

theorem complex_magnitude_fourth_power : 
  Complex.abs ((7/5 : ℂ) + (24/5 : ℂ) * Complex.I) ^ 4 = 625 := by sorry

end complex_magnitude_fourth_power_l3418_341866


namespace angela_puzzle_palace_spending_l3418_341858

/-- The amount of money Angela got to spend at Puzzle Palace -/
def total_amount : ℕ := sorry

/-- The amount of money Angela spent at Puzzle Palace -/
def amount_spent : ℕ := 78

/-- The amount of money Angela had left after shopping -/
def amount_left : ℕ := 12

/-- Theorem stating that the total amount Angela got to spend at Puzzle Palace is $90 -/
theorem angela_puzzle_palace_spending :
  total_amount = amount_spent + amount_left :=
sorry

end angela_puzzle_palace_spending_l3418_341858


namespace sisters_sandcastle_height_l3418_341864

/-- Given the height of Miki's sandcastle and the difference in height between
    the two sandcastles, calculate the height of her sister's sandcastle. -/
theorem sisters_sandcastle_height
  (miki_height : ℝ)
  (height_difference : ℝ)
  (h1 : miki_height = 0.8333333333333334)
  (h2 : height_difference = 0.3333333333333333) :
  miki_height - height_difference = 0.5 := by
sorry

end sisters_sandcastle_height_l3418_341864


namespace max_a_for_decreasing_cos_minus_sin_l3418_341899

/-- The maximum value of a for which f(x) = cos x - sin x is decreasing on [-a, a] --/
theorem max_a_for_decreasing_cos_minus_sin (a : ℝ) : 
  (∀ x ∈ Set.Icc (-a) a, 
    ∀ y ∈ Set.Icc (-a) a, 
    x < y → (Real.cos x - Real.sin x) > (Real.cos y - Real.sin y)) → 
  a ≤ π/4 :=
sorry

end max_a_for_decreasing_cos_minus_sin_l3418_341899


namespace only_hexagonal_prism_no_circular_cross_section_l3418_341849

-- Define the types of geometric shapes
inductive GeometricShape
  | Sphere
  | Cone
  | Cylinder
  | HexagonalPrism

-- Define a property for shapes that can have circular cross-sections
def has_circular_cross_section (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | GeometricShape.Cone => true
  | GeometricShape.Cylinder => true
  | GeometricShape.HexagonalPrism => false

-- Theorem statement
theorem only_hexagonal_prism_no_circular_cross_section :
  ∀ (shape : GeometricShape),
    ¬(has_circular_cross_section shape) ↔ shape = GeometricShape.HexagonalPrism :=
by
  sorry

end only_hexagonal_prism_no_circular_cross_section_l3418_341849


namespace room_occupancy_correct_answer_l3418_341892

theorem room_occupancy (num_empty_chairs : ℕ) : ℕ :=
  let total_chairs := 3 * num_empty_chairs
  let seated_people := (2 * total_chairs) / 3
  let total_people := 2 * seated_people
  total_people

theorem correct_answer : room_occupancy 8 = 32 := by
  sorry

end room_occupancy_correct_answer_l3418_341892


namespace quadratic_equation_root_l3418_341861

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = 7) ∧ 
  (3 * (-1)^2 + m * (-1) = 7) → 
  (∃ x : ℝ, x ≠ -1 ∧ 3 * x^2 + m * x = 7 ∧ x = 7/3) :=
by sorry

end quadratic_equation_root_l3418_341861


namespace divisibility_of_p_l3418_341840

theorem divisibility_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 40)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 110 < Nat.gcd s p ∧ Nat.gcd s p < 150) :
  11 ∣ p := by
  sorry

end divisibility_of_p_l3418_341840


namespace middle_group_frequency_is_32_l3418_341862

/-- Represents a frequency distribution histogram -/
structure Histogram where
  num_rectangles : ℕ
  sample_size : ℕ
  middle_rectangle_area : ℝ
  other_rectangles_area : ℝ

/-- The frequency of the middle group in the histogram -/
def middle_group_frequency (h : Histogram) : ℕ :=
  h.sample_size / 2

/-- Theorem: The frequency of the middle group is 32 under given conditions -/
theorem middle_group_frequency_is_32 (h : Histogram) 
  (h_num_rectangles : h.num_rectangles = 11)
  (h_sample_size : h.sample_size = 160)
  (h_area_equality : h.middle_rectangle_area = h.other_rectangles_area) :
  middle_group_frequency h = 32 := by
  sorry

end middle_group_frequency_is_32_l3418_341862


namespace sum_of_series_l3418_341886

/-- The sum of the infinite series ∑(n=1 to ∞) (4n+1)/3^n is equal to 7/2 -/
theorem sum_of_series : ∑' n : ℕ, (4 * n + 1 : ℝ) / 3^n = 7/2 := by
  sorry

end sum_of_series_l3418_341886


namespace effective_treatment_combination_l3418_341808

structure Treatment where
  name : String
  relieves : List String
  causes : List String

def aspirin : Treatment :=
  { name := "Aspirin"
  , relieves := ["headache", "rheumatic knee pain"]
  , causes := ["heart pain", "stomach pain"] }

def antibiotics : Treatment :=
  { name := "Antibiotics"
  , relieves := ["migraine", "heart pain"]
  , causes := ["stomach pain", "knee pain", "itching"] }

def warmCompress : Treatment :=
  { name := "Warm compress"
  , relieves := ["itching", "stomach pain"]
  , causes := [] }

def initialSymptom : String := "headache"

def isEffectiveCombination (treatments : List Treatment) : Prop :=
  (initialSymptom ∈ (treatments.bind (λ t => t.relieves))) ∧
  (∀ s, s ∈ (treatments.bind (λ t => t.causes)) →
    ∃ t ∈ treatments, s ∈ t.relieves)

theorem effective_treatment_combination :
  isEffectiveCombination [aspirin, antibiotics, warmCompress] :=
sorry

end effective_treatment_combination_l3418_341808


namespace locus_is_ellipse_l3418_341887

/-- The locus of points (x, y) in the complex plane satisfying 
    |z-2+i| + |z+3-i| = 6 is an ellipse -/
theorem locus_is_ellipse (z : ℂ) :
  let x := z.re
  let y := z.im
  (Complex.abs (z - (2 - Complex.I)) + Complex.abs (z - (-3 + Complex.I)) = 6) ↔
  ∃ (a b : ℝ) (h : 0 < b ∧ b < a),
    (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end locus_is_ellipse_l3418_341887


namespace sampling_suitable_for_yangtze_fish_yangtze_fish_sampling_only_correct_option_l3418_341844

-- Define the types of survey methods
inductive SurveyMethod
| Census
| Sampling

-- Define the scenarios
inductive Scenario
| ShellKillingRadius
| StudentHeight
| CityAirQuality
| YangtzeRiverFish

-- Define a function that determines the suitability of a survey method for a given scenario
def isSuitable (method : SurveyMethod) (scenario : Scenario) : Prop :=
  match scenario with
  | Scenario.ShellKillingRadius => method = SurveyMethod.Sampling
  | Scenario.StudentHeight => method = SurveyMethod.Census
  | Scenario.CityAirQuality => method = SurveyMethod.Sampling
  | Scenario.YangtzeRiverFish => method = SurveyMethod.Sampling

-- Theorem stating that sampling is suitable for the Yangtze River fish scenario
theorem sampling_suitable_for_yangtze_fish :
  isSuitable SurveyMethod.Sampling Scenario.YangtzeRiverFish :=
by sorry

-- Theorem stating that sampling for Yangtze River fish is the only correct option among the given scenarios
theorem yangtze_fish_sampling_only_correct_option :
  ∀ (scenario : Scenario) (method : SurveyMethod),
    (scenario = Scenario.YangtzeRiverFish ∧ method = SurveyMethod.Sampling) ↔
    (isSuitable method scenario ∧
     ((scenario = Scenario.ShellKillingRadius ∧ method = SurveyMethod.Census) ∨
      (scenario = Scenario.StudentHeight ∧ method = SurveyMethod.Sampling) ∨
      (scenario = Scenario.CityAirQuality ∧ method = SurveyMethod.Census) ∨
      (scenario = Scenario.YangtzeRiverFish ∧ method = SurveyMethod.Sampling))) :=
by sorry

end sampling_suitable_for_yangtze_fish_yangtze_fish_sampling_only_correct_option_l3418_341844


namespace cookies_per_bag_example_l3418_341883

/-- Given the number of chocolate chip cookies, oatmeal cookies, and baggies,
    calculate the number of cookies in each bag. -/
def cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) : ℕ :=
  (chocolate_chip + oatmeal) / baggies

/-- Theorem stating that with 5 chocolate chip cookies, 19 oatmeal cookies,
    and 3 baggies, there are 8 cookies in each bag. -/
theorem cookies_per_bag_example : cookies_per_bag 5 19 3 = 8 := by
  sorry

end cookies_per_bag_example_l3418_341883


namespace skateboard_bicycle_problem_l3418_341891

theorem skateboard_bicycle_problem (skateboards bicycles : ℕ) : 
  (skateboards : ℚ) / bicycles = 7 / 4 →
  skateboards = bicycles + 12 →
  skateboards + bicycles = 44 := by
sorry

end skateboard_bicycle_problem_l3418_341891


namespace cubic_equation_sum_l3418_341888

theorem cubic_equation_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a = 12 →
  b^3 - 6*b^2 + 11*b = 12 →
  c^3 - 6*c^2 + 11*c = 12 →
  a*b/c + b*c/a + c*a/b = -23/12 := by
sorry

end cubic_equation_sum_l3418_341888


namespace water_bucket_problem_l3418_341826

theorem water_bucket_problem (bucket3 bucket5 bucket6 : ℕ) 
  (h1 : bucket3 = 3)
  (h2 : bucket5 = 5)
  (h3 : bucket6 = 6) :
  bucket6 - (bucket5 - bucket3) = 4 :=
by sorry

end water_bucket_problem_l3418_341826


namespace cement_mixture_water_fraction_l3418_341877

/-- The fraction of water in a cement mixture -/
def water_fraction (total_weight sand_fraction gravel_weight : ℚ) : ℚ :=
  1 - sand_fraction - (gravel_weight / total_weight)

/-- Proof that the fraction of water in the cement mixture is 2/5 -/
theorem cement_mixture_water_fraction :
  let total_weight : ℚ := 40
  let sand_fraction : ℚ := 1/4
  let gravel_weight : ℚ := 14
  water_fraction total_weight sand_fraction gravel_weight = 2/5 := by
  sorry

end cement_mixture_water_fraction_l3418_341877


namespace vector_expression_simplification_l3418_341814

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  2 • (a - b) - 3 • (a + b) = -a - 5 • b :=
by sorry

end vector_expression_simplification_l3418_341814


namespace solve_animal_videos_problem_l3418_341897

def animal_videos_problem (cat_video_length : ℕ) : Prop :=
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  let elephant_video_length := cat_video_length + dog_video_length + gorilla_video_length
  let dolphin_video_length := cat_video_length + dog_video_length + gorilla_video_length + elephant_video_length
  let total_time := cat_video_length + dog_video_length + gorilla_video_length + elephant_video_length + dolphin_video_length
  (cat_video_length = 4) → (total_time = 144)

theorem solve_animal_videos_problem :
  animal_videos_problem 4 :=
by
  sorry

end solve_animal_videos_problem_l3418_341897


namespace circle_properties_l3418_341828

/-- A circle with center on the line y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 4)^2 = 8

/-- The line y = -4x -/
def center_line (x y : ℝ) : Prop := y = -4 * x

/-- The line x + y - 1 = 0 -/
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The point P(3, -2) -/
def point_P : ℝ × ℝ := (3, -2)

theorem circle_properties :
  ∃ (cx cy : ℝ),
    center_line cx cy ∧
    special_circle cx cy ∧
    tangent_line (point_P.1) (point_P.2) ∧
    (∀ (x y : ℝ), tangent_line x y → ((x - cx)^2 + (y - cy)^2 ≥ 8)) ∧
    ((point_P.1 - cx)^2 + (point_P.2 - cy)^2 = 8) :=
sorry

end circle_properties_l3418_341828


namespace table_movement_l3418_341890

theorem table_movement (table_width : ℝ) (table_length : ℝ) : 
  table_width = 8 ∧ table_length = 10 →
  ∃ (S : ℕ), S = 13 ∧ 
  (∀ (T : ℕ), T < S → Real.sqrt (table_width^2 + table_length^2) > T) ∧
  Real.sqrt (table_width^2 + table_length^2) ≤ S :=
by sorry

end table_movement_l3418_341890


namespace set_equality_l3418_341842

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end set_equality_l3418_341842


namespace divisor_problem_l3418_341835

theorem divisor_problem (w : ℤ) (x : ℤ) :
  (∃ k : ℤ, w = 13 * k) →
  (∃ m : ℤ, w + 3 = x * m) →
  x = 3 :=
by sorry

end divisor_problem_l3418_341835


namespace map_scale_conversion_l3418_341832

/-- Given a map scale where 15 cm represents 90 km, prove that 25 cm represents 150 km -/
theorem map_scale_conversion (scale_cm : ℝ) (scale_km : ℝ) (distance_cm : ℝ) :
  scale_cm = 15 ∧ scale_km = 90 ∧ distance_cm = 25 →
  (distance_cm / scale_cm) * scale_km = 150 := by
sorry

end map_scale_conversion_l3418_341832


namespace rectangle_area_l3418_341889

/-- Proves that the area of a rectangle with length 3 times its width and width of 4 inches is 48 square inches -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 4 →
  length = 3 * width →
  area = length * width →
  area = 48 := by
sorry

end rectangle_area_l3418_341889


namespace salmon_migration_l3418_341813

theorem salmon_migration (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261) 
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end salmon_migration_l3418_341813


namespace max_largest_integer_l3418_341868

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 50 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 11 :=
sorry

end max_largest_integer_l3418_341868


namespace range_of_x_range_of_a_l3418_341895

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

-- Part 1
theorem range_of_x (x : ℝ) (h1 : p x 1) (h2 : q x) : x ∈ Set.Ioo 2 3 := by
  sorry

-- Part 2
theorem range_of_a (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) : a ∈ Set.Ioc 1 2 := by
  sorry

end range_of_x_range_of_a_l3418_341895


namespace expression_value_l3418_341805

theorem expression_value : 
  let a : ℕ := 2017
  let b : ℕ := 2016
  let c : ℕ := 2015
  ((a^2 + b^2)^2 - c^2 - 4 * a^2 * b^2) / (a^2 + c - b^2) = 2018 := by
  sorry

end expression_value_l3418_341805


namespace cattle_breeder_milk_production_l3418_341819

/-- Calculates the weekly milk production for a given number of cows and daily milk production per cow. -/
def weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * 7

/-- Proves that 52 cows producing 1000 oz of milk per day will produce 364,000 oz of milk per week. -/
theorem cattle_breeder_milk_production :
  weekly_milk_production 52 1000 = 364000 := by
  sorry

#eval weekly_milk_production 52 1000

end cattle_breeder_milk_production_l3418_341819


namespace difference_max_min_both_l3418_341894

/-- The total number of students at the university -/
def total_students : ℕ := 2500

/-- The number of students studying German -/
def german_students : ℕ → Prop :=
  λ g => 1750 ≤ g ∧ g ≤ 1875

/-- The number of students studying Russian -/
def russian_students : ℕ → Prop :=
  λ r => 625 ≤ r ∧ r ≤ 875

/-- The number of students studying both German and Russian -/
def both_languages (g r b : ℕ) : Prop :=
  g + r - b = total_students

/-- The minimum number of students studying both languages -/
def min_both (m : ℕ) : Prop :=
  ∃ g r, german_students g ∧ russian_students r ∧ both_languages g r m ∧
  ∀ b, (∃ g' r', german_students g' ∧ russian_students r' ∧ both_languages g' r' b) → m ≤ b

/-- The maximum number of students studying both languages -/
def max_both (M : ℕ) : Prop :=
  ∃ g r, german_students g ∧ russian_students r ∧ both_languages g r M ∧
  ∀ b, (∃ g' r', german_students g' ∧ russian_students r' ∧ both_languages g' r' b) → b ≤ M

theorem difference_max_min_both :
  ∃ m M, min_both m ∧ max_both M ∧ M - m = 375 := by
  sorry

end difference_max_min_both_l3418_341894


namespace cos_pi_4_minus_alpha_l3418_341876

theorem cos_pi_4_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) :
  Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 := by
  sorry

end cos_pi_4_minus_alpha_l3418_341876


namespace f_value_at_2_f_equals_f_horner_f_2_equals_62_l3418_341822

/-- The polynomial function f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

/-- Horner's method representation of f(x) -/
def f_horner (x : ℝ) : ℝ := x*(x*(x*(2*x + 3)) + 5) - 4

theorem f_value_at_2 : f 2 = 62 := by sorry

theorem f_equals_f_horner : ∀ x, f x = f_horner x := by sorry

theorem f_2_equals_62 : f_horner 2 = 62 := by sorry

end f_value_at_2_f_equals_f_horner_f_2_equals_62_l3418_341822


namespace largest_n_for_equation_solution_exists_for_two_l3418_341845

theorem largest_n_for_equation : 
  ∀ n : ℕ+, n > 2 → 
  ¬∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12 :=
by sorry

theorem solution_exists_for_two :
  ∃ x y z : ℕ+, 2^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12 :=
by sorry

end largest_n_for_equation_solution_exists_for_two_l3418_341845


namespace minimum_formulas_to_memorize_l3418_341841

theorem minimum_formulas_to_memorize (total_formulas : ℕ) (min_score_percent : ℚ) : 
  total_formulas = 300 ∧ min_score_percent = 90 / 100 →
  ∃ (min_formulas : ℕ), 
    (min_formulas : ℚ) / total_formulas ≥ min_score_percent ∧
    ∀ (x : ℕ), (x : ℚ) / total_formulas ≥ min_score_percent → x ≥ min_formulas ∧
    min_formulas = 270 :=
by sorry

end minimum_formulas_to_memorize_l3418_341841


namespace statement_II_must_be_true_l3418_341885

-- Define the possible digits
inductive Digit
| two
| three
| five
| six
| other

-- Define the statements
def statement_I (d : Digit) : Prop := d = Digit.two
def statement_II (d : Digit) : Prop := d ≠ Digit.three
def statement_III (d : Digit) : Prop := d = Digit.five
def statement_IV (d : Digit) : Prop := d ≠ Digit.six

-- Define the problem conditions
def conditions (d : Digit) : Prop :=
  ∃ (s1 s2 s3 : Prop),
    (s1 ∧ s2 ∧ s3) ∧
    (s1 = statement_I d ∨ s1 = statement_II d ∨ s1 = statement_III d ∨ s1 = statement_IV d) ∧
    (s2 = statement_I d ∨ s2 = statement_II d ∨ s2 = statement_III d ∨ s2 = statement_IV d) ∧
    (s3 = statement_I d ∨ s3 = statement_II d ∨ s3 = statement_III d ∨ s3 = statement_IV d) ∧
    (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3)

-- Theorem: Given the conditions, Statement II must be true
theorem statement_II_must_be_true :
  ∀ d : Digit, conditions d → statement_II d :=
by
  sorry

end statement_II_must_be_true_l3418_341885


namespace ball_bearing_savings_l3418_341823

/-- Calculates the savings when buying ball bearings during a sale with bulk discount --/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (regular_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : regular_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : (num_machines * bearings_per_machine * regular_price) -
    (num_machines * bearings_per_machine * sale_price * (1 - bulk_discount)) = 120 := by
  sorry

end ball_bearing_savings_l3418_341823


namespace propositions_proof_l3418_341833

theorem propositions_proof :
  (∀ (a b c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ a * c ≤ b * d) ∧
  (∃ (a b c : ℝ), b > a ∧ a > 0 ∧ c > 0 ∧ a / b ≤ (a + c) / (b + c)) :=
by sorry

end propositions_proof_l3418_341833


namespace smallest_base_for_fraction_l3418_341867

theorem smallest_base_for_fraction (k : ℕ) : k = 14 ↔ 
  (k > 0 ∧ 
   ∀ m : ℕ, m > 0 ∧ m < k → (5 : ℚ) / 27 ≠ (m + 4 : ℚ) / (m^2 - 1) ∧
   (5 : ℚ) / 27 = (k + 4 : ℚ) / (k^2 - 1)) := by sorry

end smallest_base_for_fraction_l3418_341867


namespace inequality_solution_l3418_341847

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 0}
  else if 0 < a ∧ a < 1 then {x | (1 - Real.sqrt (1 - a^2)) / a < x ∧ x < (1 + Real.sqrt (1 - a^2)) / a}
  else if a ≥ 1 then ∅
  else if -1 < a ∧ a < 0 then {x | x > (1 - Real.sqrt (1 - a^2)) / a ∨ x < (1 + Real.sqrt (1 - a^2)) / a}
  else if a = -1 then {x | x ≠ 1 / a}
  else Set.univ

theorem inequality_solution (a : ℝ) :
  {x : ℝ | a * x^2 - 2 * x + a < 0} = solution_set a := by sorry

end inequality_solution_l3418_341847


namespace marble_arrangement_mod_1000_l3418_341860

/-- The number of blue marbles -/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that allows for a valid arrangement -/
def yellow_marbles : ℕ := 18

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of ways to arrange the marbles -/
def arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem marble_arrangement_mod_1000 :
  arrangements % 1000 = 700 := by sorry

end marble_arrangement_mod_1000_l3418_341860


namespace intersection_of_P_and_Q_l3418_341807

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | x ≤ 2} := by sorry

end intersection_of_P_and_Q_l3418_341807


namespace deepak_age_l3418_341843

theorem deepak_age (arun_age deepak_age : ℕ) : 
  arun_age / deepak_age = 2 / 3 →
  arun_age + 5 = 25 →
  deepak_age = 30 := by
sorry

end deepak_age_l3418_341843


namespace fifteenth_student_age_l3418_341853

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age : ℝ) 
  (group1_size group2_size group3_size : Nat) 
  (group1_avg group2_avg group3_avg : ℝ) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 5 →
  group2_size = 6 →
  group3_size = 3 →
  group1_avg = 13 →
  group2_avg = 15 →
  group3_avg = 17 →
  ∃ (fifteenth_student_age : ℝ),
    fifteenth_student_age = 19 ∧
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg + fifteenth_student_age) / total_students = avg_age :=
by
  sorry

end fifteenth_student_age_l3418_341853


namespace min_value_fraction_l3418_341855

theorem min_value_fraction (x : ℝ) (h : x > -1) : 
  x^2 / (x + 1) ≥ 0 ∧ ∃ y > -1, y^2 / (y + 1) = 0 := by sorry

end min_value_fraction_l3418_341855


namespace abs_y_bound_l3418_341893

theorem abs_y_bound (x y : ℝ) (h1 : |x + y| < 1/3) (h2 : |2*x - y| < 1/6) : |y| < 5/18 := by
  sorry

end abs_y_bound_l3418_341893


namespace range_of_a_for_non_negative_f_l3418_341810

/-- The range of a for which f(x) = x³ - x² - 2a has a non-negative value in (-∞, a] -/
theorem range_of_a_for_non_negative_f (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ≤ a ∧ x₀^3 - x₀^2 - 2*a ≥ 0) ↔ a ∈ Set.Icc (-1) 0 ∪ Set.Ici 2 :=
sorry

end range_of_a_for_non_negative_f_l3418_341810


namespace function_passes_through_point_l3418_341848

/-- The function f(x) = a^(x-1) + 2 passes through the point (1, 3) for any a > 0 and a ≠ 1 -/
theorem function_passes_through_point (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 2
  f 1 = 3 := by
  sorry

end function_passes_through_point_l3418_341848


namespace max_blocks_fit_l3418_341856

/-- Represents the dimensions of a rectangular box or block -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box or block given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the box and block dimensions -/
def box : Dimensions := ⟨4, 3, 2⟩
def block : Dimensions := ⟨1, 1, 2⟩

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ :=
  volume box / volume block

/-- Calculates the maximum number of blocks that can fit in the box based on physical arrangement -/
def max_blocks_by_arrangement : ℕ :=
  (box.length / block.length) * (box.width / block.width)

theorem max_blocks_fit :
  max_blocks_by_volume = 12 ∧ max_blocks_by_arrangement = 12 :=
sorry

end max_blocks_fit_l3418_341856


namespace ordered_pairs_count_l3418_341846

theorem ordered_pairs_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 32) (Finset.product (Finset.range 33) (Finset.range 33))).card ∧ n = 6 :=
by sorry

end ordered_pairs_count_l3418_341846


namespace smaller_number_in_ratio_l3418_341884

theorem smaller_number_in_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  b / a = 11 / 7 ∧
  b - a = 16 →
  a = 28 := by
sorry

end smaller_number_in_ratio_l3418_341884


namespace perspective_difference_l3418_341854

def num_students : ℕ := 250
def num_teachers : ℕ := 6
def class_sizes : List ℕ := [100, 50, 50, 25, 15, 10]

def teacher_perspective (sizes : List ℕ) : ℚ :=
  (sizes.sum : ℚ) / num_teachers

def student_perspective (sizes : List ℕ) : ℚ :=
  (sizes.map (λ x => x * x)).sum / num_students

theorem perspective_difference :
  teacher_perspective class_sizes - student_perspective class_sizes = -22.13 := by
  sorry

end perspective_difference_l3418_341854


namespace jose_profit_share_l3418_341800

/-- Calculates the share of profit for an investor given the total profit and investments --/
def calculate_profit_share (total_profit : ℚ) (investment1 : ℚ) (months1 : ℕ) (investment2 : ℚ) (months2 : ℕ) : ℚ :=
  let total_investment := investment1 * months1 + investment2 * months2
  let share_ratio := (investment2 * months2) / total_investment
  share_ratio * total_profit

/-- Proves that Jose's share of the profit is 3500 given the problem conditions --/
theorem jose_profit_share :
  let tom_investment : ℚ := 3000
  let jose_investment : ℚ := 4500
  let tom_months : ℕ := 12
  let jose_months : ℕ := 10
  let total_profit : ℚ := 6300
  calculate_profit_share total_profit tom_investment tom_months jose_investment jose_months = 3500 := by
  sorry


end jose_profit_share_l3418_341800


namespace lemonade_stand_boys_l3418_341874

theorem lemonade_stand_boys (initial_group : ℕ) : 
  let initial_boys : ℕ := (6 * initial_group) / 10
  let final_group : ℕ := initial_group
  let final_boys : ℕ := initial_boys - 3
  (6 * initial_group = 10 * initial_boys) ∧ 
  (2 * final_boys = final_group) →
  initial_boys = 18 := by
sorry

end lemonade_stand_boys_l3418_341874


namespace ball_hexagons_l3418_341863

/-- A ball made of hexagons and pentagons -/
structure Ball where
  pentagons : ℕ
  hexagons : ℕ
  pentagon_hexagon_edges : ℕ
  hexagon_pentagon_edges : ℕ

/-- Theorem: A ball with 12 pentagons has 20 hexagons -/
theorem ball_hexagons (b : Ball) 
  (h1 : b.pentagons = 12)
  (h2 : b.pentagon_hexagon_edges = 5)
  (h3 : b.hexagon_pentagon_edges = 3) :
  b.hexagons = 20 := by
  sorry

#check ball_hexagons

end ball_hexagons_l3418_341863


namespace floor_sum_example_l3418_341896

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l3418_341896


namespace sin_cos_sum_equals_sqrt2_over_2_l3418_341802

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (63 * π / 180) * Real.cos (18 * π / 180) +
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) =
  Real.sqrt 2 / 2 := by
sorry

end sin_cos_sum_equals_sqrt2_over_2_l3418_341802


namespace power_difference_quotient_l3418_341809

theorem power_difference_quotient : 
  (2^12)^2 - (2^10)^2 = 4 * ((2^11)^2 - (2^9)^2) := by
  sorry

end power_difference_quotient_l3418_341809


namespace larger_integer_value_l3418_341873

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  max a b = 21 := by
  sorry

end larger_integer_value_l3418_341873


namespace maria_journey_distance_l3418_341830

/-- A journey with two stops and a final leg -/
structure Journey where
  total_distance : ℝ
  first_stop : ℝ
  second_stop : ℝ
  final_leg : ℝ

/-- The conditions of Maria's journey -/
def maria_journey (j : Journey) : Prop :=
  j.first_stop = j.total_distance / 2 ∧
  j.second_stop = (j.total_distance - j.first_stop) / 4 ∧
  j.final_leg = 135 ∧
  j.total_distance = j.first_stop + j.second_stop + j.final_leg

/-- Theorem stating that Maria's journey has a total distance of 360 miles -/
theorem maria_journey_distance :
  ∃ j : Journey, maria_journey j ∧ j.total_distance = 360 :=
sorry

end maria_journey_distance_l3418_341830


namespace power_product_equals_five_l3418_341857

theorem power_product_equals_five (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_product_equals_five_l3418_341857


namespace point_circle_relationship_l3418_341851

theorem point_circle_relationship :
  ∀ θ : ℝ,
  let P : ℝ × ℝ := (5 * Real.cos θ, 4 * Real.sin θ)
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 25}
  P ∈ C ∨ (P.1^2 + P.2^2 < 25) :=
by sorry

end point_circle_relationship_l3418_341851


namespace sqrt_8_same_type_as_sqrt_2_l3418_341812

-- Define what it means for two square roots to be of the same type
def same_type (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b

-- State the theorem
theorem sqrt_8_same_type_as_sqrt_2 :
  same_type (Real.sqrt 8) (Real.sqrt 2) :=
by
  sorry

end sqrt_8_same_type_as_sqrt_2_l3418_341812


namespace tom_lake_crossing_cost_l3418_341839

/-- The cost of hiring an assistant for crossing a lake back and forth -/
def lake_crossing_cost (one_way_time : ℕ) (hourly_rate : ℕ) : ℕ :=
  2 * one_way_time * hourly_rate

/-- Theorem: The cost for Tom to hire an assistant for crossing the lake back and forth is $80 -/
theorem tom_lake_crossing_cost :
  lake_crossing_cost 4 10 = 80 := by
  sorry

end tom_lake_crossing_cost_l3418_341839


namespace rabbits_ate_three_watermelons_l3418_341831

/-- The number of watermelons eaten by rabbits -/
def watermelons_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem: Given that Sam initially grew 4 watermelons and now has 1 left,
    prove that rabbits ate 3 watermelons -/
theorem rabbits_ate_three_watermelons :
  watermelons_eaten 4 1 = 3 := by
  sorry

end rabbits_ate_three_watermelons_l3418_341831


namespace min_value_expression_min_value_achievable_l3418_341869

theorem min_value_expression (y : ℝ) (h : y > 2) :
  (y^2 + y + 1) / Real.sqrt (y - 2) ≥ 3 * Real.sqrt 35 :=
by sorry

theorem min_value_achievable :
  ∃ y : ℝ, y > 2 ∧ (y^2 + y + 1) / Real.sqrt (y - 2) = 3 * Real.sqrt 35 :=
by sorry

end min_value_expression_min_value_achievable_l3418_341869


namespace quadratic_function_m_values_l3418_341880

theorem quadratic_function_m_values (m : ℝ) :
  (∃ a b c : ℝ, ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = a * x^2 + b * x + c) →
  (m = 3 ∨ m = -1) ∧
  ((m = 3 → ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = 6 * x^2 + 9) ∧
   (m = -1 → ∀ x, (m^2 - m) * x^(m^2 - 2*m - 1) + (m - 3) * x + m^2 = 2 * x^2 - 4 * x + 1)) :=
by sorry

end quadratic_function_m_values_l3418_341880


namespace complex_division_result_l3418_341829

theorem complex_division_result : 
  let i := Complex.I
  (3 + i) / (1 + i) = 2 - i := by sorry

end complex_division_result_l3418_341829


namespace table_arrangement_l3418_341850

theorem table_arrangement (total_tables : Nat) (num_rows : Nat) 
  (tables_per_row : Nat) (leftover : Nat) : 
  total_tables = 74 → num_rows = 8 → 
  tables_per_row = total_tables / num_rows →
  leftover = total_tables % num_rows →
  tables_per_row = 9 ∧ leftover = 2 := by
  sorry

end table_arrangement_l3418_341850


namespace johns_computer_cost_l3418_341821

/-- The total cost of John's computer setup -/
def total_cost (computer_cost peripherals_cost original_video_card_cost upgraded_video_card_cost : ℝ) : ℝ :=
  computer_cost + peripherals_cost + (upgraded_video_card_cost - original_video_card_cost)

/-- Theorem stating the total cost of John's computer setup -/
theorem johns_computer_cost :
  let computer_cost : ℝ := 1500
  let peripherals_cost : ℝ := computer_cost / 5
  let original_video_card_cost : ℝ := 300
  let upgraded_video_card_cost : ℝ := 2 * original_video_card_cost
  total_cost computer_cost peripherals_cost original_video_card_cost upgraded_video_card_cost = 2100 :=
by
  sorry

end johns_computer_cost_l3418_341821


namespace student_contribution_l3418_341881

theorem student_contribution 
  (total_raised : ℕ) 
  (num_students : ℕ) 
  (cost_per_student : ℕ) 
  (remaining_funds : ℕ) : 
  total_raised = 50 → 
  num_students = 20 → 
  cost_per_student = 7 → 
  remaining_funds = 10 → 
  (total_raised - remaining_funds) / num_students = 5 :=
by sorry

end student_contribution_l3418_341881


namespace fixed_point_of_exponential_function_l3418_341870

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ 3 - a^(x + 1)
  f (-1) = 2 := by sorry

end fixed_point_of_exponential_function_l3418_341870
