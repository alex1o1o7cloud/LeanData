import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1424_142434

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.tan θ = 1/6) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1424_142434


namespace NUMINAMATH_CALUDE_temperature_decrease_l1424_142444

theorem temperature_decrease (initial_temp final_temp decrease : ℤ) :
  initial_temp = -3 →
  decrease = 6 →
  final_temp = initial_temp - decrease →
  final_temp = -9 :=
by sorry

end NUMINAMATH_CALUDE_temperature_decrease_l1424_142444


namespace NUMINAMATH_CALUDE_sin_theta_value_l1424_142476

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) 
  (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : Real.sin θ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1424_142476


namespace NUMINAMATH_CALUDE_square_difference_characterization_l1424_142477

theorem square_difference_characterization (N : ℕ+) :
  (∃ k : ℕ, (2^N.val : ℕ) - 2 * N.val = k^2) ↔ N = 1 ∨ N = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_characterization_l1424_142477


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1424_142485

/-- The speed of a boat in still water, given its downstream and upstream distances traveled in one hour. -/
theorem boat_speed_in_still_water (downstream upstream : ℝ) (h1 : downstream = 11) (h2 : upstream = 5) :
  let boat_speed := (downstream + upstream) / 2
  boat_speed = 8 := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1424_142485


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l1424_142422

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if a list of natural numbers contains distinct elements -/
def isDistinct (list : List ℕ) : Prop := list.Nodup

/-- The theorem stating that 43 is the smallest prime that is the sum of five distinct primes -/
theorem smallest_prime_sum_of_five_primes :
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ),
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ ∧ isPrime p₅ ∧
    isDistinct [p₁, p₂, p₃, p₄, p₅] ∧
    p₁ + p₂ + p₃ + p₄ + p₅ = 43 ∧
    isPrime 43 ∧
    (∀ (q : ℕ), q < 43 →
      ¬∃ (q₁ q₂ q₃ q₄ q₅ : ℕ),
        isPrime q₁ ∧ isPrime q₂ ∧ isPrime q₃ ∧ isPrime q₄ ∧ isPrime q₅ ∧
        isDistinct [q₁, q₂, q₃, q₄, q₅] ∧
        q₁ + q₂ + q₃ + q₄ + q₅ = q ∧
        isPrime q) :=
by sorry


end NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l1424_142422


namespace NUMINAMATH_CALUDE_question_differentiates_inhabitants_l1424_142468

-- Define the types of inhabitants
inductive InhabitantType
  | TruthTeller
  | Liar

-- Define the possible answers
inductive Answer
  | Yes
  | No

-- Function to determine how an inhabitant would answer the question
def answer_question (inhabitant_type : InhabitantType) : Answer :=
  match inhabitant_type with
  | InhabitantType.TruthTeller => Answer.No
  | InhabitantType.Liar => Answer.Yes

-- Theorem stating that the question can differentiate between truth-tellers and liars
theorem question_differentiates_inhabitants :
  ∀ (t : InhabitantType),
    (t = InhabitantType.TruthTeller ↔ answer_question t = Answer.No) ∧
    (t = InhabitantType.Liar ↔ answer_question t = Answer.Yes) :=
by sorry

end NUMINAMATH_CALUDE_question_differentiates_inhabitants_l1424_142468


namespace NUMINAMATH_CALUDE_flour_sugar_difference_l1424_142413

/-- Given a recipe and current ingredients, calculate the difference between additional flour needed and total sugar needed. -/
theorem flour_sugar_difference 
  (total_flour : ℕ) 
  (total_sugar : ℕ) 
  (added_flour : ℕ) 
  (h1 : total_flour = 10) 
  (h2 : total_sugar = 2) 
  (h3 : added_flour = 7) : 
  (total_flour - added_flour) - total_sugar = 1 := by
  sorry

#check flour_sugar_difference

end NUMINAMATH_CALUDE_flour_sugar_difference_l1424_142413


namespace NUMINAMATH_CALUDE_laura_drives_234_miles_per_week_l1424_142453

/-- Calculates the total miles driven per week based on Laura's travel habits -/
def total_miles_per_week (school_round_trip : ℕ) (supermarket_extra : ℕ) (gym_distance : ℕ) (friend_distance : ℕ) : ℕ :=
  let school_miles := school_round_trip * 5
  let supermarket_miles := (school_round_trip + 2 * supermarket_extra) * 2
  let gym_miles := 2 * gym_distance * 3
  let friend_miles := 2 * friend_distance
  school_miles + supermarket_miles + gym_miles + friend_miles

/-- Theorem stating that Laura drives 234 miles per week -/
theorem laura_drives_234_miles_per_week :
  total_miles_per_week 20 10 5 12 = 234 := by
  sorry

end NUMINAMATH_CALUDE_laura_drives_234_miles_per_week_l1424_142453


namespace NUMINAMATH_CALUDE_factorization_equality_l1424_142488

theorem factorization_equality (x y : ℝ) : -4*x^2 + 12*x*y - 9*y^2 = -(2*x - 3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1424_142488


namespace NUMINAMATH_CALUDE_tony_distance_behind_l1424_142441

-- Define the slope length
def slope_length : ℝ := 700

-- Define the meeting point distance from the top
def meeting_point : ℝ := 70

-- Define Maria's and Tony's uphill speeds as variables
variable (maria_uphill_speed tony_uphill_speed : ℝ)

-- Define the theorem
theorem tony_distance_behind (maria_uphill_speed tony_uphill_speed : ℝ) 
  (h_positive : maria_uphill_speed > 0 ∧ tony_uphill_speed > 0) :
  let maria_total_distance := slope_length + slope_length / 2
  let tony_total_distance := maria_total_distance * (tony_uphill_speed / maria_uphill_speed)
  let distance_behind := maria_total_distance - tony_total_distance
  2 * distance_behind = 300 := by sorry

end NUMINAMATH_CALUDE_tony_distance_behind_l1424_142441


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1424_142469

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, -2) and (-4, 10) is 6. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := -2
  let x₂ : ℝ := -4
  let y₂ : ℝ := 10
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 6 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1424_142469


namespace NUMINAMATH_CALUDE_mixture_theorem_l1424_142489

/-- Represents a mixture of three liquids -/
structure Mixture where
  lemon : ℚ
  oil : ℚ
  vinegar : ℚ

/-- Mix A composition -/
def mixA : Mixture := ⟨1, 2, 3⟩

/-- Mix B composition -/
def mixB : Mixture := ⟨3, 4, 5⟩

/-- Checks if it's possible to create a target mixture from Mix A and Mix B -/
def canCreateMixture (target : Mixture) : Prop :=
  ∃ (x y : ℚ), x ≥ 0 ∧ y ≥ 0 ∧
    x * mixA.lemon + y * mixB.lemon = (x + y) * target.lemon ∧
    x * mixA.oil + y * mixB.oil = (x + y) * target.oil ∧
    x * mixA.vinegar + y * mixB.vinegar = (x + y) * target.vinegar

theorem mixture_theorem :
  canCreateMixture ⟨3, 5, 7⟩ ∧
  ¬canCreateMixture ⟨2, 5, 8⟩ ∧
  ¬canCreateMixture ⟨4, 5, 6⟩ ∧
  ¬canCreateMixture ⟨5, 6, 7⟩ := by
  sorry


end NUMINAMATH_CALUDE_mixture_theorem_l1424_142489


namespace NUMINAMATH_CALUDE_units_digit_pow_two_cycle_units_digit_pow_two_2015_l1424_142403

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_pow_two_cycle (n : ℕ) (h : n ≥ 1) : 
  units_digit (2^n) = units_digit (2^((n - 1) % 4 + 1)) :=
sorry

theorem units_digit_pow_two_2015 : units_digit (2^2015) = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_pow_two_cycle_units_digit_pow_two_2015_l1424_142403


namespace NUMINAMATH_CALUDE_no_common_root_for_rational_coefficients_l1424_142464

theorem no_common_root_for_rational_coefficients :
  ∀ (a b : ℚ), ¬∃ (x : ℂ), (x^5 - x - 1 = 0) ∧ (x^2 + a*x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_for_rational_coefficients_l1424_142464


namespace NUMINAMATH_CALUDE_joint_account_final_amount_l1424_142448

/-- Calculates the final amount in a joint account after one year with changing interest rates and tax --/
theorem joint_account_final_amount 
  (deposit_lopez : ℝ) 
  (deposit_johnson : ℝ) 
  (initial_rate : ℝ) 
  (changed_rate : ℝ) 
  (tax_rate : ℝ) 
  (h1 : deposit_lopez = 100)
  (h2 : deposit_johnson = 150)
  (h3 : initial_rate = 0.20)
  (h4 : changed_rate = 0.18)
  (h5 : tax_rate = 0.05) : 
  ∃ (final_amount : ℝ), abs (final_amount - 272.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_joint_account_final_amount_l1424_142448


namespace NUMINAMATH_CALUDE_sector_radius_l1424_142465

/-- Given a circular sector with area 240π and arc length 20π, prove that its radius is 24. -/
theorem sector_radius (A : ℝ) (L : ℝ) (r : ℝ) : 
  A = 240 * Real.pi → L = 20 * Real.pi → A = (1/2) * r^2 * (L/r) → r = 24 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l1424_142465


namespace NUMINAMATH_CALUDE_imaginary_iff_m_neq_zero_pure_imaginary_iff_m_eq_two_or_three_not_in_second_quadrant_l1424_142455

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := (m^2 - 5*m + 6) - 3*m*Complex.I

-- Theorem 1: z is an imaginary number iff m ≠ 0
theorem imaginary_iff_m_neq_zero (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m ≠ 0 :=
sorry

-- Theorem 2: z is a pure imaginary number iff m = 2 or m = 3
theorem pure_imaginary_iff_m_eq_two_or_three (m : ℝ) :
  (z m).re = 0 ↔ m = 2 ∨ m = 3 :=
sorry

-- Theorem 3: z cannot be in the second quadrant for any real m
theorem not_in_second_quadrant (m : ℝ) :
  ¬((z m).re < 0 ∧ (z m).im > 0) :=
sorry

end NUMINAMATH_CALUDE_imaginary_iff_m_neq_zero_pure_imaginary_iff_m_eq_two_or_three_not_in_second_quadrant_l1424_142455


namespace NUMINAMATH_CALUDE_equation_solution_l1424_142410

theorem equation_solution :
  ∃ (a b c d : ℝ), 
    2 * a^2 + b^2 + 2 * c^2 + 2 = 3 * d + Real.sqrt (2 * a + b + 2 * c - 3 * d) ∧
    d = 2/3 ∧ a = 1/2 ∧ b = 1 ∧ c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1424_142410


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_ratio_max_l1424_142450

theorem rectangle_area_perimeter_ratio_max (A P : ℝ) (h1 : A > 0) (h2 : P > 0) : 
  A / P^2 ≤ 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_ratio_max_l1424_142450


namespace NUMINAMATH_CALUDE_child_workers_count_l1424_142409

/-- Represents the number of child workers employed by the contractor. -/
def num_child_workers : ℕ := 5

/-- Represents the number of male workers employed by the contractor. -/
def num_male_workers : ℕ := 20

/-- Represents the number of female workers employed by the contractor. -/
def num_female_workers : ℕ := 15

/-- Represents the daily wage of a male worker in rupees. -/
def male_wage : ℕ := 25

/-- Represents the daily wage of a female worker in rupees. -/
def female_wage : ℕ := 20

/-- Represents the daily wage of a child worker in rupees. -/
def child_wage : ℕ := 8

/-- Represents the average daily wage paid by the contractor in rupees. -/
def average_wage : ℕ := 21

/-- Theorem stating that the number of child workers is 5, given the conditions. -/
theorem child_workers_count :
  (num_male_workers * male_wage + num_female_workers * female_wage + num_child_workers * child_wage) / 
  (num_male_workers + num_female_workers + num_child_workers) = average_wage := by
  sorry

end NUMINAMATH_CALUDE_child_workers_count_l1424_142409


namespace NUMINAMATH_CALUDE_sequence_with_geometric_differences_formula_l1424_142429

def sequence_with_geometric_differences (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = (1/3)^(n-1)

theorem sequence_with_geometric_differences_formula (a : ℕ → ℝ) :
  sequence_with_geometric_differences a →
  ∀ n : ℕ, n ≥ 1 → a n = 3/2 * (1 - (1/3)^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_with_geometric_differences_formula_l1424_142429


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l1424_142402

theorem jerrys_average_increase :
  ∀ (original_average new_average : ℚ),
  original_average = 94 →
  (3 * original_average + 102) / 4 = new_average →
  new_average - original_average = 2 := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l1424_142402


namespace NUMINAMATH_CALUDE_vowel_probability_is_three_thirteenths_l1424_142473

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The set of vowels including W -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'W'}

/-- The probability of selecting a vowel from the alphabet -/
def vowel_probability : ℚ := (Finset.card vowels : ℚ) / alphabet_size

theorem vowel_probability_is_three_thirteenths : 
  vowel_probability = 3 / 13 := by sorry

end NUMINAMATH_CALUDE_vowel_probability_is_three_thirteenths_l1424_142473


namespace NUMINAMATH_CALUDE_magnitude_of_c_l1424_142480

/-- Given vectors a and b, with c parallel to b and its projection onto a being 2, 
    prove that the magnitude of c is 2√5 -/
theorem magnitude_of_c (a b c : ℝ × ℝ) : 
  a = (1, 0) → 
  b = (1, 2) → 
  (c.1 / b.1 = c.2 / b.2) →  -- c is parallel to b
  (c.1 * a.1 + c.2 * a.2) / Real.sqrt (a.1^2 + a.2^2) = 2 →  -- projection of c onto a is 2
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_c_l1424_142480


namespace NUMINAMATH_CALUDE_jerrie_carrie_difference_l1424_142430

/-- The number of sit-ups Barney can perform in one minute -/
def barney_rate : ℕ := 45

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_rate : ℕ := 2 * barney_rate

/-- The number of minutes Barney performs sit-ups -/
def barney_time : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_time : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_time : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 510

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_rate : ℕ := (total_situps - (barney_rate * barney_time + carrie_rate * carrie_time)) / jerrie_time

theorem jerrie_carrie_difference :
  jerrie_rate - carrie_rate = 5 :=
sorry

end NUMINAMATH_CALUDE_jerrie_carrie_difference_l1424_142430


namespace NUMINAMATH_CALUDE_roots_equal_magnitude_opposite_sign_l1424_142420

theorem roots_equal_magnitude_opposite_sign (a b c m : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = -x ∧
    (x^2 - b*x) / (a*x - c) = (m - 1) / (m + 1) ∧
    (y^2 - b*y) / (a*y - c) = (m - 1) / (m + 1)) →
  m = (a - b) / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_roots_equal_magnitude_opposite_sign_l1424_142420


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l1424_142486

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 120) : 
  x^2 + y^2 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l1424_142486


namespace NUMINAMATH_CALUDE_power_tower_mod_2000_l1424_142484

theorem power_tower_mod_2000 : 
  (5 : ℕ) ^ (5 ^ (5 ^ 5)) ≡ 625 [MOD 2000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_2000_l1424_142484


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1424_142414

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x - 24) → (∃ y : ℝ, y^2 = 10*y - 24 ∧ x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1424_142414


namespace NUMINAMATH_CALUDE_minimum_words_to_learn_l1424_142495

theorem minimum_words_to_learn (total_words : ℕ) (required_percentage : ℚ) : 
  total_words = 600 → required_percentage = 90 / 100 → 
  ∃ (min_words : ℕ), min_words * 100 ≥ total_words * required_percentage ∧
    ∀ (n : ℕ), n * 100 ≥ total_words * required_percentage → n ≥ min_words :=
by sorry

end NUMINAMATH_CALUDE_minimum_words_to_learn_l1424_142495


namespace NUMINAMATH_CALUDE_stratified_sampling_l1424_142491

theorem stratified_sampling (total_families : ℕ) (high_income : ℕ) (middle_income : ℕ) (low_income : ℕ) 
  (high_income_sampled : ℕ) (h1 : total_families = 500) (h2 : high_income = 125) (h3 : middle_income = 280) 
  (h4 : low_income = 95) (h5 : high_income_sampled = 25) :
  (high_income_sampled * low_income) / high_income = 19 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1424_142491


namespace NUMINAMATH_CALUDE_arrangements_count_l1424_142408

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- The number of elderly persons --/
def num_elderly : ℕ := 1

/-- The total number of people --/
def total_people : ℕ := num_volunteers + num_elderly

/-- The position of the elderly person --/
def elderly_position : ℕ := (total_people + 1) / 2

theorem arrangements_count :
  (num_volunteers.factorial * (num_volunteers + 1 - elderly_position).factorial) = 24 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l1424_142408


namespace NUMINAMATH_CALUDE_min_exposed_surface_area_l1424_142443

/- Define a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ
  volume_eq : length * width * height = 128
  positive : length > 0 ∧ width > 0 ∧ height > 0

/- Define the three solids -/
def solid1 : RectangularSolid := {
  length := 4,
  width := 1,
  height := 32,
  volume_eq := by norm_num,
  positive := by simp
}

def solid2 : RectangularSolid := {
  length := 8,
  width := 8,
  height := 2,
  volume_eq := by norm_num,
  positive := by simp
}

def solid3 : RectangularSolid := {
  length := 4,
  width := 2,
  height := 16,
  volume_eq := by norm_num,
  positive := by simp
}

/- Calculate the exposed surface area of the tower -/
def exposedSurfaceArea (s1 s2 s3 : RectangularSolid) : ℝ :=
  2 * (s1.length * s1.width + s2.length * s2.width + s3.length * s3.width) +
  2 * (s1.length * s1.height + s2.length * s2.height + s3.length * s3.height) +
  2 * (s1.width * s1.height + s2.width * s2.height + s3.width * s3.height) -
  2 * (s1.length * s1.width + s2.length * s2.width)

/- Theorem statement -/
theorem min_exposed_surface_area :
  exposedSurfaceArea solid1 solid2 solid3 = 832 := by sorry

end NUMINAMATH_CALUDE_min_exposed_surface_area_l1424_142443


namespace NUMINAMATH_CALUDE_cube_surface_area_for_given_volume_l1424_142492

def cube_volume : ℝ := 3375

def cube_surface_area (v : ℝ) : ℝ :=
  6 * (v ^ (1/3)) ^ 2

theorem cube_surface_area_for_given_volume :
  cube_surface_area cube_volume = 1350 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_for_given_volume_l1424_142492


namespace NUMINAMATH_CALUDE_max_discount_percentage_l1424_142460

theorem max_discount_percentage (cost : ℝ) (price : ℝ) (min_margin : ℝ) :
  cost = 400 →
  price = 500 →
  min_margin = 0.0625 →
  ∃ x : ℝ, x = 15 ∧
    ∀ y : ℝ, 0 ≤ y → y ≤ x →
      price * (1 - y / 100) - cost ≥ cost * min_margin ∧
      ∀ z : ℝ, z > x →
        price * (1 - z / 100) - cost < cost * min_margin :=
by sorry

end NUMINAMATH_CALUDE_max_discount_percentage_l1424_142460


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l1424_142400

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l1424_142400


namespace NUMINAMATH_CALUDE_installment_value_approximation_l1424_142421

def tv_price : ℕ := 15000
def num_installments : ℕ := 20
def interest_rate : ℚ := 6 / 100
def last_installment : ℕ := 13000

def calculate_installment_value (price : ℕ) (num_inst : ℕ) (rate : ℚ) (last_inst : ℕ) : ℚ :=
  let avg_balance : ℚ := price / 2
  let interest : ℚ := avg_balance * rate
  let total_amount : ℚ := price + interest
  (total_amount - last_inst) / (num_inst - 1)

theorem installment_value_approximation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_installment_value tv_price num_installments interest_rate last_installment - 129| < ε :=
sorry

end NUMINAMATH_CALUDE_installment_value_approximation_l1424_142421


namespace NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l1424_142474

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ ≥ 3 * Real.sqrt 2 :=
by sorry

theorem equality_condition (θ : Real) (h : θ = π / 4) :
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l1424_142474


namespace NUMINAMATH_CALUDE_triangle_angle_tangent_difference_l1424_142440

theorem triangle_angle_tangent_difference (A B : Real) (cosA tanB : Real) 
  (h1 : cosA = -Real.sqrt 2 / 2)
  (h2 : tanB = 1 / 3) :
  Real.tan (A - B) = -2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_tangent_difference_l1424_142440


namespace NUMINAMATH_CALUDE_primitive_root_extension_l1424_142462

theorem primitive_root_extension (p : ℕ) (x : ℤ) (h_p : Nat.Prime p) (h_p_odd : p % 2 = 1)
  (h_primitive_root_p2 : IsPrimitiveRoot x (p^2)) :
  ∀ α : ℕ, α ≥ 2 → IsPrimitiveRoot x (p^α) :=
by sorry

end NUMINAMATH_CALUDE_primitive_root_extension_l1424_142462


namespace NUMINAMATH_CALUDE_correct_tax_distribution_l1424_142435

-- Define the types of taxes
inductive TaxType
  | PropertyTax
  | FederalTax
  | ProfitTax
  | RegionalTax
  | TransportTax

-- Define the budget levels
inductive BudgetLevel
  | Federal
  | Regional

-- Function to map tax types to budget levels
def taxDistribution : TaxType → BudgetLevel
  | TaxType.PropertyTax => BudgetLevel.Regional
  | TaxType.FederalTax => BudgetLevel.Federal
  | TaxType.ProfitTax => BudgetLevel.Regional
  | TaxType.RegionalTax => BudgetLevel.Regional
  | TaxType.TransportTax => BudgetLevel.Regional

-- Theorem stating the correct distribution of taxes
theorem correct_tax_distribution :
  (taxDistribution TaxType.PropertyTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.FederalTax = BudgetLevel.Federal) ∧
  (taxDistribution TaxType.ProfitTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.RegionalTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.TransportTax = BudgetLevel.Regional) :=
by sorry

end NUMINAMATH_CALUDE_correct_tax_distribution_l1424_142435


namespace NUMINAMATH_CALUDE_batting_bowling_average_change_l1424_142431

/-- Represents a batsman's performance in a cricket inning -/
structure InningPerformance where
  runs : ℕ
  boundaries : ℕ
  sixes : ℕ
  strike_rate : ℝ
  wickets : ℕ

/-- Calculates the new batting average after an inning -/
def new_batting_average (initial_average : ℝ) (performance : InningPerformance) : ℝ :=
  initial_average + 5

/-- Calculates the new bowling average after an inning -/
def new_bowling_average (initial_average : ℝ) (performance : InningPerformance) : ℝ :=
  initial_average - 3

theorem batting_bowling_average_change 
  (A B : ℝ) 
  (performance : InningPerformance) 
  (h1 : performance.runs = 100) 
  (h2 : performance.boundaries = 12) 
  (h3 : performance.sixes = 2) 
  (h4 : performance.strike_rate = 130) 
  (h5 : performance.wickets = 1) :
  new_batting_average A performance = A + 5 ∧ 
  new_bowling_average B performance = B - 3 := by
  sorry


end NUMINAMATH_CALUDE_batting_bowling_average_change_l1424_142431


namespace NUMINAMATH_CALUDE_solve_euro_equation_l1424_142456

-- Define the operation €
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem solve_euro_equation (x : ℝ) :
  (euro 6 (euro x 5) = 480) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_euro_equation_l1424_142456


namespace NUMINAMATH_CALUDE_last_four_average_l1424_142438

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 63.75 :=
by sorry

end NUMINAMATH_CALUDE_last_four_average_l1424_142438


namespace NUMINAMATH_CALUDE_fencing_cost_l1424_142483

/-- Given a rectangular field with sides in ratio 3:4 and area 10092 sq. m,
    prove that the cost of fencing at 25 paise per metre is 101.5 rupees. -/
theorem fencing_cost (length width : ℝ) (h1 : length / width = 3 / 4)
  (h2 : length * width = 10092) : 
  (2 * (length + width) * 25 / 100) = 101.5 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_l1424_142483


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l1424_142412

/-- The equation of a line passing through a chord of an ellipse -/
theorem chord_equation_of_ellipse (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 2) →         -- Midpoint x-coordinate is 2
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  ∃ (x y : ℝ), x + 4*y - 10 = 0  -- Equation of the line
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l1424_142412


namespace NUMINAMATH_CALUDE_peters_horses_food_l1424_142425

/-- The amount of food needed for Peter's horses over 5 days -/
def food_needed (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) 
                (grain_per_meal : ℕ) (grain_meals_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_horses * (oats_per_meal * oats_meals_per_day + grain_per_meal * grain_meals_per_day) * num_days

/-- Theorem stating the total amount of food needed for Peter's horses -/
theorem peters_horses_food : 
  food_needed 6 5 3 4 2 5 = 690 := by
  sorry

end NUMINAMATH_CALUDE_peters_horses_food_l1424_142425


namespace NUMINAMATH_CALUDE_probability_sum_binary_digits_not_exceed_eight_l1424_142447

/-- The maximum number in the set of possible values -/
def max_num : ℕ := 2016

/-- Function to calculate the sum of binary digits of a natural number -/
def sum_binary_digits (n : ℕ) : ℕ := sorry

/-- The count of numbers from 1 to max_num with sum of binary digits not exceeding 8 -/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the probability of a randomly chosen number from 1 to max_num 
    having a sum of binary digits not exceeding 8 -/
theorem probability_sum_binary_digits_not_exceed_eight :
  (count_valid_numbers : ℚ) / max_num = 655 / 672 := by sorry

end NUMINAMATH_CALUDE_probability_sum_binary_digits_not_exceed_eight_l1424_142447


namespace NUMINAMATH_CALUDE_total_money_proof_l1424_142433

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
sorry

end NUMINAMATH_CALUDE_total_money_proof_l1424_142433


namespace NUMINAMATH_CALUDE_paper_tearing_theorem_l1424_142401

/-- Represents the number of parts after n tears -/
def num_parts (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that the number of parts is always odd and can never be 100 -/
theorem paper_tearing_theorem :
  ∀ n : ℕ, ∃ k : ℕ, num_parts n = 2 * k + 1 ∧ num_parts n ≠ 100 :=
sorry

end NUMINAMATH_CALUDE_paper_tearing_theorem_l1424_142401


namespace NUMINAMATH_CALUDE_problems_solved_l1424_142452

theorem problems_solved (first last : ℕ) (h : first = 78 ∧ last = 125) : 
  (last - first + 1 : ℕ) = 49 := by
  sorry

end NUMINAMATH_CALUDE_problems_solved_l1424_142452


namespace NUMINAMATH_CALUDE_eight_distinct_lengths_l1424_142487

/-- Represents an isosceles right triangle with side length 24 -/
structure IsoscelesRightTriangle :=
  (side : ℝ)
  (is_24 : side = 24)

/-- Counts the number of distinct integer lengths of line segments from a vertex to the hypotenuse -/
def count_distinct_integer_lengths (t : IsoscelesRightTriangle) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 8 distinct integer lengths -/
theorem eight_distinct_lengths (t : IsoscelesRightTriangle) : 
  count_distinct_integer_lengths t = 8 := by sorry

end NUMINAMATH_CALUDE_eight_distinct_lengths_l1424_142487


namespace NUMINAMATH_CALUDE_f_properties_l1424_142472

def f (x : ℝ) := (x - 2)^2

theorem f_properties :
  (∀ x, f (x + 2) = f (-x - 2)) ∧
  (∀ x < 2, ∀ y < x, f y < f x) ∧
  (∀ x > 2, ∀ y > x, f y > f x) ∧
  (∀ x y, x < y → f (y + 2) - f y > f (x + 2) - f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1424_142472


namespace NUMINAMATH_CALUDE_fraction_equality_l1424_142459

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 4) :
  (4 * a + 3 * b) / (4 * a - 3 * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1424_142459


namespace NUMINAMATH_CALUDE_C_power_50_l1424_142416

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_power_50 : C^50 = !![(-299 : ℤ), -100; 800, 251] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1424_142416


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_parametric_quadratic_inequality_solution_l1424_142466

-- Define the quadratic function
def f (x : ℝ) := -x^2 - 2*x + 3

-- Define the parametric quadratic function
def g (a : ℝ) (x : ℝ) := -x^2 - 2*x + a

theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

theorem parametric_quadratic_inequality_solution (a : ℝ) :
  ({x : ℝ | g a x < 0} = Set.univ) ↔ a < -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_parametric_quadratic_inequality_solution_l1424_142466


namespace NUMINAMATH_CALUDE_first_class_students_l1424_142428

theorem first_class_students (x : ℕ) : 
  (∃ (total_students : ℕ),
    total_students = x + 50 ∧
    (50 * x + 60 * 50 : ℚ) / total_students = 56.25) →
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_first_class_students_l1424_142428


namespace NUMINAMATH_CALUDE_cube_order_equivalence_l1424_142445

theorem cube_order_equivalence (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_order_equivalence_l1424_142445


namespace NUMINAMATH_CALUDE_oil_distribution_l1424_142457

theorem oil_distribution (a b c : ℝ) : 
  c = 48 →
  (2/3 * a = 4/5 * (b + 1/3 * a)) →
  (2/3 * a = 48 + 1/5 * (b + 1/3 * a)) →
  a = 96 ∧ b = 48 := by
sorry

end NUMINAMATH_CALUDE_oil_distribution_l1424_142457


namespace NUMINAMATH_CALUDE_min_divisions_is_48_l1424_142467

/-- Represents a cell division strategy -/
structure DivisionStrategy where
  div42 : ℕ  -- number of divisions resulting in 42 cells
  div44 : ℕ  -- number of divisions resulting in 44 cells

/-- The number of cells after applying a division strategy -/
def resultingCells (s : DivisionStrategy) : ℕ :=
  1 + 41 * s.div42 + 43 * s.div44

/-- A division strategy is valid if it results in exactly 1993 cells -/
def isValidStrategy (s : DivisionStrategy) : Prop :=
  resultingCells s = 1993

/-- The total number of divisions in a strategy -/
def totalDivisions (s : DivisionStrategy) : ℕ :=
  s.div42 + s.div44

/-- There exists a valid division strategy -/
axiom exists_valid_strategy : ∃ s : DivisionStrategy, isValidStrategy s

/-- The minimum number of divisions needed is 48 -/
theorem min_divisions_is_48 :
  ∃ s : DivisionStrategy, isValidStrategy s ∧
    totalDivisions s = 48 ∧
    ∀ t : DivisionStrategy, isValidStrategy t → totalDivisions s ≤ totalDivisions t :=
  sorry

end NUMINAMATH_CALUDE_min_divisions_is_48_l1424_142467


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1424_142461

theorem quadratic_factorization (a : ℝ) : a^2 - 8*a + 16 = (a - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1424_142461


namespace NUMINAMATH_CALUDE_f_2019_equals_2016_l1424_142470

def f : ℕ → ℕ
| x => if x ≤ 2015 then x + 2 else f (x - 5)

theorem f_2019_equals_2016 : f 2019 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_equals_2016_l1424_142470


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1424_142426

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def real_axis_length (a : ℝ) : ℝ := 2 * a

def imaginary_axis_length (b : ℝ) : ℝ := 2 * b

def asymptote_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : real_axis_length a = 2 * Real.sqrt 2)
  (h4 : imaginary_axis_length b = 2) :
  ∀ (x y : ℝ), asymptote_equation a b x y ↔ y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1424_142426


namespace NUMINAMATH_CALUDE_problem_polygon_area_l1424_142463

-- Define a point on a 2D grid
structure GridPoint where
  x : Int
  y : Int

-- Define a polygon as a list of grid points
def Polygon := List GridPoint

-- Function to calculate the area of a polygon given its vertices
def polygonArea (p : Polygon) : ℚ :=
  sorry

-- Define the specific polygon from the problem
def problemPolygon : Polygon := [
  ⟨0, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨30, 0⟩, ⟨40, 0⟩, ⟨40, 10⟩,
  ⟨40, 20⟩, ⟨30, 30⟩, ⟨20, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨0, 10⟩
]

-- Theorem statement
theorem problem_polygon_area :
  polygonArea problemPolygon = 15/2 := by sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l1424_142463


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1424_142490

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1424_142490


namespace NUMINAMATH_CALUDE_total_glasses_displayed_l1424_142406

/-- Represents the number of cupboards of each type -/
def num_tall_cupboards : ℕ := 2
def num_wide_cupboards : ℕ := 2
def num_narrow_cupboards : ℕ := 2

/-- Represents the capacity of each type of cupboard -/
def tall_cupboard_capacity : ℕ := 30
def wide_cupboard_capacity : ℕ := 2 * tall_cupboard_capacity
def narrow_cupboard_capacity : ℕ := 45

/-- Represents the number of shelves in a narrow cupboard -/
def shelves_per_narrow_cupboard : ℕ := 3

/-- Represents the number of broken shelves -/
def broken_shelves : ℕ := 1

/-- Theorem stating the total number of glasses displayed -/
theorem total_glasses_displayed : 
  num_tall_cupboards * tall_cupboard_capacity +
  num_wide_cupboards * wide_cupboard_capacity +
  (num_narrow_cupboards * narrow_cupboard_capacity - 
   broken_shelves * (narrow_cupboard_capacity / shelves_per_narrow_cupboard)) = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_glasses_displayed_l1424_142406


namespace NUMINAMATH_CALUDE_zach_allowance_is_five_l1424_142493

/-- Calculates Zach's weekly allowance given the conditions of his savings and earnings -/
def zachsAllowance (bikeCost lawnMowingPay babysittingRatePerHour babysittingHours currentSavings additionalNeeded : ℕ) : ℕ :=
  let totalNeeded := bikeCost - additionalNeeded
  let remainingToEarn := totalNeeded - currentSavings
  let otherEarnings := lawnMowingPay + babysittingRatePerHour * babysittingHours
  remainingToEarn - otherEarnings

/-- Proves that Zach's weekly allowance is $5 given the specified conditions -/
theorem zach_allowance_is_five :
  zachsAllowance 100 10 7 2 65 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_zach_allowance_is_five_l1424_142493


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_z_l1424_142424

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem max_imaginary_part_of_z (z : ℂ) 
  (h : is_purely_imaginary ((z - 6) / (z - 8*I))) : 
  (⨆ (z : ℂ), |z.im|) = 9 := by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_z_l1424_142424


namespace NUMINAMATH_CALUDE_smallest_square_area_for_radius_6_l1424_142499

/-- The area of the smallest square that can contain a circle with a given radius -/
def smallest_square_area (radius : ℝ) : ℝ :=
  (2 * radius) ^ 2

/-- Theorem: The area of the smallest square that can contain a circle with a radius of 6 is 144 -/
theorem smallest_square_area_for_radius_6 :
  smallest_square_area 6 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_radius_6_l1424_142499


namespace NUMINAMATH_CALUDE_binary_divisible_by_seven_l1424_142458

def K (x y z : Fin 2) : ℕ :=
  524288 + 131072 + 65536 + 16384 + 4096 + 1024 + 256 + 64 * y.val + 32 * x.val + 16 * z.val + 8 + 2

theorem binary_divisible_by_seven (x y z : Fin 2) :
  K x y z % 7 = 0 → x = 0 ∧ y = 1 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_binary_divisible_by_seven_l1424_142458


namespace NUMINAMATH_CALUDE_probability_same_color_is_240_11970_l1424_142482

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The number of marbles drawn -/
def marbles_drawn : ℕ := 4

/-- The probability of drawing four marbles of the same color -/
def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_is_240_11970 :
  probability_same_color = 240 / 11970 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_240_11970_l1424_142482


namespace NUMINAMATH_CALUDE_selection_ways_l1424_142475

/-- The number of students in the group -/
def num_students : ℕ := 5

/-- The number of positions to be filled (representative and vice-president) -/
def num_positions : ℕ := 2

/-- Theorem: The number of ways to select one representative and one vice-president
    from a group of 5 students is equal to 20 -/
theorem selection_ways : (num_students * (num_students - 1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l1424_142475


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1424_142423

theorem reciprocal_of_negative_fraction (a b : ℚ) (h : b ≠ 0) :
  ((-a) / b)⁻¹ = -(b / a) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l1424_142423


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l1424_142479

theorem sum_of_four_integers (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25 →
  a + b + c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l1424_142479


namespace NUMINAMATH_CALUDE_square_difference_ends_with_two_l1424_142417

theorem square_difference_ends_with_two (a b : ℕ) (h1 : a^2 > b^2) 
  (h2 : ∃ (m n : ℕ), a^2 = m^2 ∧ b^2 = n^2) 
  (h3 : (a^2 - b^2) % 10 = 2) :
  a^2 % 10 = 6 ∧ b^2 % 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_difference_ends_with_two_l1424_142417


namespace NUMINAMATH_CALUDE_area_of_right_isosceles_triangle_l1424_142439

/-- A right-angled isosceles triangle with the sum of the areas of squares on its sides equal to 72 -/
structure RightIsoscelesTriangle where
  /-- The length of each of the two equal sides -/
  side : ℝ
  /-- The sum of the areas of squares on the sides is 72 -/
  sum_of_squares : side^2 + side^2 + (2 * side^2) = 72

/-- The area of a right-angled isosceles triangle with the given property is 9 -/
theorem area_of_right_isosceles_triangle (t : RightIsoscelesTriangle) : 
  (1/2 : ℝ) * t.side * t.side = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_isosceles_triangle_l1424_142439


namespace NUMINAMATH_CALUDE_b_squared_neq_ac_sufficient_not_necessary_l1424_142437

-- Define what it means for three numbers to form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b / a = c / b) ∨ (a = 0 ∧ b = 0) ∨ (b = 0 ∧ c = 0)

-- State the theorem
theorem b_squared_neq_ac_sufficient_not_necessary :
  (∀ a b c : ℝ, b^2 ≠ a*c → ¬is_geometric_sequence a b c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_sequence a b c) := by sorry

end NUMINAMATH_CALUDE_b_squared_neq_ac_sufficient_not_necessary_l1424_142437


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l1424_142497

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ colored_lipstick : ℕ, colored_lipstick = total_students / 2)
  (h3 : ∃ red_lipstick : ℕ, red_lipstick = colored_lipstick / 4)
  (h4 : ∃ blue_lipstick : ℕ, blue_lipstick = red_lipstick / 5) :
  blue_lipstick = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l1424_142497


namespace NUMINAMATH_CALUDE_sum_of_sequences_l1424_142471

/-- Sum of arithmetic sequence with 5 terms -/
def arithmetic_sum (a₁ : ℕ) : ℕ := a₁ + (a₁ + 10) + (a₁ + 20) + (a₁ + 30) + (a₁ + 40)

/-- The sum of two specific arithmetic sequences equals 270 -/
theorem sum_of_sequences : arithmetic_sum 3 + arithmetic_sum 11 = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l1424_142471


namespace NUMINAMATH_CALUDE_tire_circumference_l1424_142449

/-- Calculates the circumference of a car tire given the car's speed and tire rotation rate. -/
theorem tire_circumference (speed : ℝ) (rotations : ℝ) : 
  speed = 168 → rotations = 400 → (speed * 1000 / 60) / rotations = 7 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l1424_142449


namespace NUMINAMATH_CALUDE_gcd_equality_pairs_l1424_142418

theorem gcd_equality_pairs :
  ∀ a b : ℕ+, a ≤ b →
  (∀ x : ℕ+, Nat.gcd x a * Nat.gcd x b = Nat.gcd x 20 * Nat.gcd x 22) →
  ((a = 2 ∧ b = 220) ∨ (a = 4 ∧ b = 110) ∨ (a = 10 ∧ b = 44) ∨ (a = 20 ∧ b = 22)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_equality_pairs_l1424_142418


namespace NUMINAMATH_CALUDE_ninth_term_is_negative_256_l1424_142442

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℤ, a (n + 1) = a n * q
  prod_condition : a 2 * a 5 = -32
  sum_condition : a 3 + a 4 = 4

/-- The theorem stating that a₉ = -256 for the given geometric sequence -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_negative_256_l1424_142442


namespace NUMINAMATH_CALUDE_soccer_ball_white_patches_l1424_142404

/-- Represents a soccer ball with hexagonal and pentagonal patches -/
structure SoccerBall where
  total_patches : ℕ
  white_patches : ℕ
  black_patches : ℕ
  white_black_borders : ℕ

/-- Conditions for a valid soccer ball configuration -/
def is_valid_soccer_ball (ball : SoccerBall) : Prop :=
  ball.total_patches = 32 ∧
  ball.white_patches + ball.black_patches = ball.total_patches ∧
  ball.white_black_borders = 3 * ball.white_patches ∧
  ball.white_black_borders = 5 * ball.black_patches

/-- Theorem stating that a valid soccer ball has 20 white patches -/
theorem soccer_ball_white_patches (ball : SoccerBall) 
  (h : is_valid_soccer_ball ball) : ball.white_patches = 20 := by
  sorry

#check soccer_ball_white_patches

end NUMINAMATH_CALUDE_soccer_ball_white_patches_l1424_142404


namespace NUMINAMATH_CALUDE_min_value_product_l1424_142494

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (a₀ + 3 * b₀) * (b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1424_142494


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_max_nonnegative_l1424_142419

-- Problem 1
theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

-- Problem 2
theorem max_nonnegative (x : ℝ) :
  let a := x^2 - 1
  let b := 2*x + 2
  max a b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_max_nonnegative_l1424_142419


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l1424_142407

/-- The parabola function y = x^2 - 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 1

/-- The line function y = r -/
def line (r : ℝ) (x : ℝ) : ℝ := r

/-- The area of the triangle formed by the vertex of the parabola and its intersections with the line y = r -/
def triangleArea (r : ℝ) : ℝ := (r + 1)^(3/2)

theorem triangle_area_bounds (r : ℝ) :
  (8 ≤ triangleArea r ∧ triangleArea r ≤ 64) → (3 ≤ r ∧ r ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l1424_142407


namespace NUMINAMATH_CALUDE_tens_digit_of_2031_pow_2024_minus_2033_l1424_142451

theorem tens_digit_of_2031_pow_2024_minus_2033 :
  ∃ n : ℕ, n < 10 ∧ (2031^2024 - 2033) % 100 = 80 + n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2031_pow_2024_minus_2033_l1424_142451


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l1424_142436

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

/-- Theorem: Given the son's age is 22 and the age difference is 24,
    the number of years until the man's age is twice his son's age is 2 -/
theorem double_age_in_two_years :
  years_until_double_age 22 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l1424_142436


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_six_l1424_142496

theorem no_solution_iff_m_eq_neg_six (m : ℝ) :
  (∀ x : ℝ, x ≠ -2 → (x - 3) / (x + 2) + (x + 1) / (x + 2) ≠ m / (x + 2)) ↔ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_six_l1424_142496


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1424_142415

theorem inequality_solution_sets 
  (a b : ℝ) 
  (h1 : ∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1424_142415


namespace NUMINAMATH_CALUDE_remainder_theorem_l1424_142478

/-- Given a polynomial f(x) with the following properties:
    1) When divided by (x-1), the remainder is 8
    2) When divided by (x+1), the remainder is 1
    This theorem states that the remainder when f(x) is divided by (x^2-1) is -7x-9 -/
theorem remainder_theorem (f : ℝ → ℝ) 
  (h1 : ∃ g : ℝ → ℝ, ∀ x, f x = g x * (x - 1) + 8)
  (h2 : ∃ h : ℝ → ℝ, ∀ x, f x = h x * (x + 1) + 1) :
  ∃ q : ℝ → ℝ, ∀ x, f x = q x * (x^2 - 1) + (-7*x - 9) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1424_142478


namespace NUMINAMATH_CALUDE_sports_club_membership_l1424_142446

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 80 →
  badminton = 48 →
  tennis = 46 →
  neither = 7 →
  badminton + tennis - (total - neither) = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l1424_142446


namespace NUMINAMATH_CALUDE_weight_of_seven_moles_l1424_142498

/-- The weight of a given number of moles of a compound -/
def weight_of_moles (molecular_weight : ℕ) (moles : ℕ) : ℕ :=
  molecular_weight * moles

/-- Theorem: The weight of 7 moles of a compound with molecular weight 2856 is 19992 -/
theorem weight_of_seven_moles :
  weight_of_moles 2856 7 = 19992 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_seven_moles_l1424_142498


namespace NUMINAMATH_CALUDE_angle_c_possibilities_l1424_142481

theorem angle_c_possibilities : ∃ (s : Finset ℕ), 
  (∀ c ∈ s, ∃ d : ℕ, 
    c > 0 ∧ d > 0 ∧ 
    c + d = 180 ∧ 
    ∃ k : ℕ, k > 0 ∧ c = k * d) ∧
  (∀ c : ℕ, 
    (∃ d : ℕ, c > 0 ∧ d > 0 ∧ c + d = 180 ∧ ∃ k : ℕ, k > 0 ∧ c = k * d) → 
    c ∈ s) ∧
  s.card = 17 :=
sorry

end NUMINAMATH_CALUDE_angle_c_possibilities_l1424_142481


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1424_142411

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 4^5 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1424_142411


namespace NUMINAMATH_CALUDE_product_abcd_l1424_142405

theorem product_abcd (a b c d : ℚ) 
  (eq1 : 4*a + 5*b + 7*c + 9*d = 82)
  (eq2 : d + c = 2*b)
  (eq3 : 2*b + 2*c = 3*a)
  (eq4 : c - 2 = d) :
  a * b * c * d = 276264960 / 14747943 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l1424_142405


namespace NUMINAMATH_CALUDE_factorial_square_root_square_l1424_142432

-- Definition of factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_square_root_square :
  (((factorial 5 + 1) * factorial 4).sqrt ^ 2 : ℕ) = 2904 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_square_l1424_142432


namespace NUMINAMATH_CALUDE_equation_solution_l1424_142454

theorem equation_solution : 
  ∃ x : ℝ, ((x * 5) / 2.5) - (8 * 2.25) = 5.5 ∧ x = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1424_142454


namespace NUMINAMATH_CALUDE_largest_integer_quadratic_inequality_six_satisfies_inequality_seven_does_not_satisfy_inequality_l1424_142427

theorem largest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 9*n + 14 < 0 → n ≤ 6 :=
by
  sorry

theorem six_satisfies_inequality :
  (6 : ℤ)^2 - 9*6 + 14 < 0 :=
by
  sorry

theorem seven_does_not_satisfy_inequality :
  (7 : ℤ)^2 - 9*7 + 14 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_quadratic_inequality_six_satisfies_inequality_seven_does_not_satisfy_inequality_l1424_142427
