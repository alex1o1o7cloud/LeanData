import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_number_proof_l1153_115343

theorem two_digit_number_proof : 
  ∀ n : ℕ, 
  (10 ≤ n ∧ n < 100) → -- two-digit number
  (∃ x y : ℕ, n = 10 * x + y ∧ y = x + 3 ∧ n = y * y) → -- conditions
  (n = 25 ∨ n = 36) := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l1153_115343


namespace NUMINAMATH_CALUDE_distribute_18_balls_5_boxes_l1153_115376

/-- The number of ways to distribute n identical balls into k distinct boxes,
    with each box containing at least m balls. -/
def distribute_balls (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

theorem distribute_18_balls_5_boxes :
  distribute_balls 18 5 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_distribute_18_balls_5_boxes_l1153_115376


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l1153_115361

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 10 →
  capacity_ratio = 2 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 25 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l1153_115361


namespace NUMINAMATH_CALUDE_tan_alpha_3_expression_equals_2_l1153_115374

theorem tan_alpha_3_expression_equals_2 (α : Real) (h : Real.tan α = 3) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_expression_equals_2_l1153_115374


namespace NUMINAMATH_CALUDE_complex_magnitude_l1153_115393

theorem complex_magnitude (z : ℂ) (h1 : z.im = 2) (h2 : (z^2 + 3).re = 0) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1153_115393


namespace NUMINAMATH_CALUDE_jonathan_distance_l1153_115395

theorem jonathan_distance (J : ℝ) 
  (mercedes_distance : ℝ → ℝ)
  (davonte_distance : ℝ → ℝ)
  (h1 : mercedes_distance J = 2 * J)
  (h2 : davonte_distance J = mercedes_distance J + 2)
  (h3 : mercedes_distance J + davonte_distance J = 32) :
  J = 7.5 := by
sorry

end NUMINAMATH_CALUDE_jonathan_distance_l1153_115395


namespace NUMINAMATH_CALUDE_total_balloons_l1153_115359

theorem total_balloons (tom_balloons sara_balloons : ℕ) 
  (h1 : tom_balloons = 9) 
  (h2 : sara_balloons = 8) : 
  tom_balloons + sara_balloons = 17 := by
sorry

end NUMINAMATH_CALUDE_total_balloons_l1153_115359


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1153_115316

theorem arithmetic_progression_first_term
  (n : ℕ)
  (a_n : ℝ)
  (d : ℝ)
  (h1 : n = 15)
  (h2 : a_n = 44)
  (h3 : d = 3) :
  ∃ a₁ : ℝ, a₁ = 2 ∧ a_n = a₁ + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1153_115316


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1153_115364

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 27 years older than his son and the son's present age is 25 years. -/
theorem man_son_age_ratio :
  let son_age : ℕ := 25
  let man_age : ℕ := son_age + 27
  let son_age_in_two_years : ℕ := son_age + 2
  let man_age_in_two_years : ℕ := man_age + 2
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l1153_115364


namespace NUMINAMATH_CALUDE_original_flock_size_l1153_115377

/-- Represents the flock size and its changes over time -/
structure FlockDynamics where
  initialSize : ℕ
  yearlyKilled : ℕ
  yearlyBorn : ℕ
  years : ℕ
  joinedFlockSize : ℕ
  finalCombinedSize : ℕ

/-- Theorem stating the original flock size given the conditions -/
theorem original_flock_size (fd : FlockDynamics)
  (h1 : fd.yearlyKilled = 20)
  (h2 : fd.yearlyBorn = 30)
  (h3 : fd.years = 5)
  (h4 : fd.joinedFlockSize = 150)
  (h5 : fd.finalCombinedSize = 300)
  : fd.initialSize = 100 := by
  sorry

#check original_flock_size

end NUMINAMATH_CALUDE_original_flock_size_l1153_115377


namespace NUMINAMATH_CALUDE_quadratic_comparison_l1153_115335

/-- Given two quadratic functions A and B, prove that B can be expressed in terms of x
    and that A is always greater than B for all real x. -/
theorem quadratic_comparison (x : ℝ) : 
  let A := 3 * x^2 - 2 * x + 1
  let B := 2 * x^2 - x - 3
  (A + B = 5 * x^2 - 4 * x - 2) → (A > B) := by sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l1153_115335


namespace NUMINAMATH_CALUDE_prob_less_than_8_l1153_115333

/-- The probability of scoring less than 8 in a single shot, given the probabilities of hitting the 10, 9, and 8 rings. -/
theorem prob_less_than_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  1 - p10 - p9 - p8 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_8_l1153_115333


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l1153_115341

theorem inequality_range_theorem (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, |2 - x| + |x + 1| ≤ a) ↔ a ∈ Set.Ici 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l1153_115341


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l1153_115307

def num_candidates : ℕ := 4

theorem president_vice_president_selection :
  (num_candidates * (num_candidates - 1) = 12) := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l1153_115307


namespace NUMINAMATH_CALUDE_triangle_side_length_l1153_115351

/-- Given a triangle XYZ with sides x, y, and z, where y = 7, z = 6, and cos(Y - Z) = 47/64,
    prove that x = √63.75 -/
theorem triangle_side_length (x y z : ℝ) (Y Z : ℝ) :
  y = 7 →
  z = 6 →
  Real.cos (Y - Z) = 47 / 64 →
  x = Real.sqrt 63.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1153_115351


namespace NUMINAMATH_CALUDE_survey_analysis_l1153_115331

/-- Represents the survey data and population information -/
structure SurveyData where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  surveyed_male : ℕ
  surveyed_female : ℕ
  male_enthusiasts : ℕ
  male_non_enthusiasts : ℕ
  female_enthusiasts : ℕ
  female_non_enthusiasts : ℕ

/-- Calculates the K² value for the chi-square test -/
def calculate_k_squared (data : SurveyData) : ℚ :=
  let n := data.surveyed_male + data.surveyed_female
  let a := data.male_enthusiasts
  let b := data.male_non_enthusiasts
  let c := data.female_enthusiasts
  let d := data.female_non_enthusiasts
  (n : ℚ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The main theorem to prove -/
theorem survey_analysis (data : SurveyData) 
    (h1 : data.total_students = 9000)
    (h2 : data.male_students = 4000)
    (h3 : data.female_students = 5000)
    (h4 : data.surveyed_male = 40)
    (h5 : data.surveyed_female = 50)
    (h6 : data.male_enthusiasts = 20)
    (h7 : data.male_non_enthusiasts = 20)
    (h8 : data.female_enthusiasts = 40)
    (h9 : data.female_non_enthusiasts = 10) :
    (data.surveyed_male : ℚ) / data.surveyed_female = data.male_students / data.female_students ∧
    calculate_k_squared data > 6635 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_survey_analysis_l1153_115331


namespace NUMINAMATH_CALUDE_desk_rearrangement_combinations_l1153_115390

theorem desk_rearrangement_combinations : 
  let day1_choices : ℕ := 1
  let day2_choices : ℕ := 2
  let day3_choices : ℕ := 3
  let day4_choices : ℕ := 2
  let day5_choices : ℕ := 1
  day1_choices * day2_choices * day3_choices * day4_choices * day5_choices = 12 := by
sorry

end NUMINAMATH_CALUDE_desk_rearrangement_combinations_l1153_115390


namespace NUMINAMATH_CALUDE_even_function_extension_l1153_115340

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function for x < 0
def f_neg (x : ℝ) : ℝ := x * (2 * x - 1)

-- Theorem statement
theorem even_function_extension 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_neg : ∀ x, x < 0 → f x = f_neg x) : 
  ∀ x, x > 0 → f x = x * (2 * x + 1) := by
sorry


end NUMINAMATH_CALUDE_even_function_extension_l1153_115340


namespace NUMINAMATH_CALUDE_third_median_length_l1153_115306

/-- An isosceles triangle with specific median lengths and area -/
structure SpecialIsoscelesTriangle where
  -- Two sides of equal length
  base : ℝ
  leg : ℝ
  -- Two medians of equal length
  equalMedian : ℝ
  -- The third median
  thirdMedian : ℝ
  -- Constraints
  isIsosceles : base > 0 ∧ leg > 0
  equalMedianLength : equalMedian = 4
  areaConstraint : area = 3 * Real.sqrt 15
  -- Area calculation (placeholder)
  area : ℝ := sorry

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : SpecialIsoscelesTriangle) : 
  t.thirdMedian = 2 * Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_third_median_length_l1153_115306


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_one_l1153_115398

theorem negative_three_less_than_negative_one : -3 < -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_one_l1153_115398


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1153_115347

theorem inequality_implies_upper_bound (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1153_115347


namespace NUMINAMATH_CALUDE_victor_books_left_l1153_115353

def book_count (initial bought gifted donated : ℕ) : ℕ :=
  initial + bought - gifted - donated

theorem victor_books_left : book_count 25 12 7 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_victor_books_left_l1153_115353


namespace NUMINAMATH_CALUDE_bug_return_probability_l1153_115332

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - P n)

/-- The probability of the bug returning to its starting vertex after 12 moves -/
theorem bug_return_probability : P 12 = 14762 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l1153_115332


namespace NUMINAMATH_CALUDE_daycare_toddlers_l1153_115360

/-- Given a day care center with toddlers and infants, prove that under certain conditions, 
    the number of toddlers is 42. -/
theorem daycare_toddlers (T I : ℕ) : 
  T / I = 7 / 3 →  -- Initial ratio of toddlers to infants
  T / (I + 12) = 7 / 5 →  -- New ratio after 12 infants join
  T = 42 := by
  sorry

end NUMINAMATH_CALUDE_daycare_toddlers_l1153_115360


namespace NUMINAMATH_CALUDE_odd_sum_probability_l1153_115356

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 1

def odd_sum_pairs : Finset (ℕ × ℕ) :=
  (cards.product cards).filter (λ pair => pair.1 < pair.2 ∧ is_odd_sum pair)

theorem odd_sum_probability :
  (odd_sum_pairs.card : ℚ) / (cards.card.choose 2) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l1153_115356


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1153_115367

theorem committee_selection_ways (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 5) :
  Nat.choose n k = 118755 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1153_115367


namespace NUMINAMATH_CALUDE_sum_of_five_terms_positive_l1153_115348

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isMonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b → f b ≤ f a

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_five_terms_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf_odd : isOddFunction f)
  (hf_mono : ∀ x y, 0 ≤ x → 0 ≤ y → isMonotonicallyDecreasing f x y)
  (ha_arith : isArithmeticSequence a)
  (ha3_neg : a 3 < 0) :
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_terms_positive_l1153_115348


namespace NUMINAMATH_CALUDE_log_product_equality_l1153_115330

theorem log_product_equality : Real.log 3 / Real.log 8 * (Real.log 32 / Real.log 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l1153_115330


namespace NUMINAMATH_CALUDE_trajectory_of_P_l1153_115368

-- Define the coordinate system
variable (O : ℝ × ℝ)  -- Origin
variable (A B P : ℝ → ℝ × ℝ)  -- Points as functions of time

-- Define the conditions
axiom origin : O = (0, 0)
axiom A_on_x_axis : ∀ t, (A t).2 = 0
axiom B_on_y_axis : ∀ t, (B t).1 = 0
axiom AB_length : ∀ t, Real.sqrt ((A t).1^2 + (B t).2^2) = 3
axiom P_position : ∀ t, P t = (2/3 • A t) + (1/3 • B t)

-- State the theorem
theorem trajectory_of_P :
  ∀ t, (P t).1^2 / 4 + (P t).2^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l1153_115368


namespace NUMINAMATH_CALUDE_remainder_415_420_mod_16_l1153_115315

theorem remainder_415_420_mod_16 : 415^420 ≡ 1 [MOD 16] := by
  sorry

end NUMINAMATH_CALUDE_remainder_415_420_mod_16_l1153_115315


namespace NUMINAMATH_CALUDE_feet_heads_difference_l1153_115303

theorem feet_heads_difference : 
  let birds : ℕ := 4
  let dogs : ℕ := 3
  let cats : ℕ := 18
  let humans : ℕ := 7
  let bird_feet : ℕ := 2
  let dog_feet : ℕ := 4
  let cat_feet : ℕ := 4
  let human_feet : ℕ := 2
  let total_heads : ℕ := birds + dogs + cats + humans
  let total_feet : ℕ := birds * bird_feet + dogs * dog_feet + cats * cat_feet + humans * human_feet
  total_feet - total_heads = 74 :=
by sorry

end NUMINAMATH_CALUDE_feet_heads_difference_l1153_115303


namespace NUMINAMATH_CALUDE_square_graph_triangles_l1153_115338

/-- A planar graph formed by a square with interior points -/
structure SquareGraph where
  /-- The number of interior points in the square -/
  interior_points : ℕ
  /-- The total number of vertices in the graph -/
  vertices : ℕ
  /-- The total number of edges in the graph -/
  edges : ℕ
  /-- The total number of faces in the graph (including the exterior face) -/
  faces : ℕ
  /-- The condition that the graph is formed by a square with interior points -/
  square_condition : vertices = interior_points + 4
  /-- The condition that all regions except the exterior are triangles -/
  triangle_condition : 2 * edges = 3 * (faces - 1) + 4
  /-- Euler's formula for planar graphs -/
  euler_formula : vertices - edges + faces = 2

/-- The theorem stating the number of triangles in the specific square graph -/
theorem square_graph_triangles (g : SquareGraph) (h : g.interior_points = 20) :
  g.faces - 1 = 42 := by
  sorry

end NUMINAMATH_CALUDE_square_graph_triangles_l1153_115338


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1153_115362

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -4.5 < x ∧ x < 3.5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1153_115362


namespace NUMINAMATH_CALUDE_scallops_per_person_is_two_l1153_115396

-- Define the constants from the problem
def scallops_per_pound : ℕ := 8
def cost_per_pound : ℚ := 24
def number_of_people : ℕ := 8
def total_cost : ℚ := 48

-- Define the function to calculate scallops per person
def scallops_per_person : ℚ :=
  (total_cost / cost_per_pound * scallops_per_pound) / number_of_people

-- Theorem to prove
theorem scallops_per_person_is_two : scallops_per_person = 2 := by
  sorry

end NUMINAMATH_CALUDE_scallops_per_person_is_two_l1153_115396


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l1153_115380

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.cos (α - β) = -(59/72) := by sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l1153_115380


namespace NUMINAMATH_CALUDE_sum_lent_is_300_l1153_115391

/-- Proves that the sum lent is 300, given the conditions of the problem -/
theorem sum_lent_is_300 
  (interest_rate : ℝ) 
  (loan_duration : ℕ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.04)
  (h2 : loan_duration = 8)
  (h3 : interest_difference = 204) :
  ∃ (principal : ℝ), 
    principal * interest_rate * loan_duration = principal - interest_difference ∧ 
    principal = 300 := by
sorry


end NUMINAMATH_CALUDE_sum_lent_is_300_l1153_115391


namespace NUMINAMATH_CALUDE_election_ratio_l1153_115381

theorem election_ratio (X Y : ℝ) 
  (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) 
  (h2 : X > 0) 
  (h3 : Y > 0) : 
  X / Y = 2 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l1153_115381


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l1153_115385

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * Real.sqrt ((x - 1)^2 + y^2) = 2 * (x + 1)

-- Define the trajectory equation
def trajectory_equation (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y^2 = 4 * x

-- Define the intersection points M and N
def intersection_points (m : ℝ) (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  y₁ = x₁ + m ∧ y₂ = x₂ + m ∧
  trajectory_equation M ∧ trajectory_equation N ∧
  m ≠ 0

-- Define the perpendicularity condition
def perpendicular (O M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem trajectory_and_intersection :
  (∀ P, P_condition P → trajectory_equation P) ∧
  (∀ m M N, intersection_points m M N → perpendicular (0, 0) M N → m = -4) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l1153_115385


namespace NUMINAMATH_CALUDE_problem_statement_l1153_115372

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (x + y + z) * (1/x + 1/y + 1/z) = 91/10) :
  ⌊(x^3 + y^3 + z^3) * (1/x^3 + 1/y^3 + 1/z^3)⌋ = 9 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1153_115372


namespace NUMINAMATH_CALUDE_product_of_exponents_l1153_115300

theorem product_of_exponents (p r s : ℕ) : 
  (2^p + 2^3 = 18) → 
  (3^r + 3 = 30) → 
  (4^s + 4^2 = 276) → 
  p * r * s = 48 := by
sorry

end NUMINAMATH_CALUDE_product_of_exponents_l1153_115300


namespace NUMINAMATH_CALUDE_percentage_equality_l1153_115370

theorem percentage_equality (x : ℝ) (h : x = 130) : 
  (65 / 100 * x) / 422.50 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1153_115370


namespace NUMINAMATH_CALUDE_unique_function_solution_l1153_115334

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) → 
  (∀ x : ℝ, f x = x) := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l1153_115334


namespace NUMINAMATH_CALUDE_carmen_candle_usage_l1153_115320

/-- Calculates the number of candles used given the burning time per night and total nights -/
def candles_used (burn_time_per_night : ℚ) (total_nights : ℕ) : ℚ :=
  (burn_time_per_night * total_nights) / 8

theorem carmen_candle_usage :
  candles_used 2 24 = 6 := by sorry

end NUMINAMATH_CALUDE_carmen_candle_usage_l1153_115320


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1153_115375

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1153_115375


namespace NUMINAMATH_CALUDE_periodic_odd_function_value_l1153_115327

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_value
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 6)
  (h_odd : is_odd f)
  (h_value : f (-1) = -1) :
  f 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_value_l1153_115327


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l1153_115339

/-- Given a quadratic polynomial x^2 - sx + q with roots a and b, 
    where a + b = a^2 + b^2 = a^3 + b^3 = ... = a^2008 + b^2008,
    the maximum value of 1/a^2009 + 1/b^2009 is 2. -/
theorem max_reciprocal_sum (s q a b : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2008 → a^n + b^n = a + b) →
  a * b = q →
  a + b = s →
  x^2 - s*x + q = (x - a) * (x - b) →
  (∃ M : ℝ, ∀ s' q' a' b' : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2008 → a'^n + b'^n = a' + b') →
    a' * b' = q' →
    a' + b' = s' →
    x^2 - s'*x + q' = (x - a') * (x - b') →
    1 / a'^2009 + 1 / b'^2009 ≤ M) ∧
  1 / a^2009 + 1 / b^2009 = M →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l1153_115339


namespace NUMINAMATH_CALUDE_quadratic_properties_l1153_115371

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 4*x + 3

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem quadratic_properties :
  (∀ x ∈ domain, f (-x + 4) = f x) ∧  -- Axis of symmetry at x = 2
  (f 2 = 7) ∧  -- Vertex at (2, 7)
  (∀ x ∈ domain, f x ≤ 7) ∧  -- Maximum value
  (∀ x ∈ domain, f x ≥ 6) ∧  -- Minimum value
  (∃ x ∈ domain, f x = 7) ∧  -- Maximum is attained
  (∃ x ∈ domain, f x = 6) :=  -- Minimum is attained
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1153_115371


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_l1153_115318

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the interior angles of a triangle
def interior_angles (t : Triangle) : ℝ := sorry

-- Theorem: The sum of interior angles of a triangle is 180°
theorem sum_of_interior_angles (t : Triangle) : interior_angles t = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_l1153_115318


namespace NUMINAMATH_CALUDE_vovochka_max_candies_l1153_115350

/-- Represents the problem of distributing candies among classmates -/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies Vovochka can keep -/
def max_candies_kept (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates - 1) * (cd.min_group_candies / cd.min_group_size)

/-- Theorem stating the maximum number of candies Vovochka can keep -/
theorem vovochka_max_candies :
  let cd : CandyDistribution := {
    total_candies := 200,
    num_classmates := 25,
    min_group_size := 16,
    min_group_candies := 100
  }
  max_candies_kept cd = 37 := by
  sorry

end NUMINAMATH_CALUDE_vovochka_max_candies_l1153_115350


namespace NUMINAMATH_CALUDE_min_weeks_for_puppies_l1153_115363

/-- Represents the different types of puppies Bob can buy -/
inductive PuppyType
  | GoldenRetriever
  | Poodle
  | Beagle

/-- Calculates the minimum number of weeks Bob needs to compete to afford a puppy -/
def min_weeks_to_afford (puppy : PuppyType) (entrance_fee : ℕ) (first_place_prize : ℕ) (current_savings : ℕ) : ℕ :=
  match puppy with
  | PuppyType.GoldenRetriever => 10
  | PuppyType.Poodle => 7
  | PuppyType.Beagle => 5

/-- Theorem stating the minimum number of weeks Bob needs to compete for each puppy type -/
theorem min_weeks_for_puppies 
  (entrance_fee : ℕ) 
  (first_place_prize : ℕ) 
  (second_place_prize : ℕ) 
  (third_place_prize : ℕ) 
  (current_savings : ℕ) 
  (golden_price : ℕ) 
  (poodle_price : ℕ) 
  (beagle_price : ℕ) 
  (h1 : entrance_fee = 10)
  (h2 : first_place_prize = 100)
  (h3 : second_place_prize = 70)
  (h4 : third_place_prize = 40)
  (h5 : current_savings = 180)
  (h6 : golden_price = 1000)
  (h7 : poodle_price = 800)
  (h8 : beagle_price = 600) :
  (min_weeks_to_afford PuppyType.GoldenRetriever entrance_fee first_place_prize current_savings = 10) ∧
  (min_weeks_to_afford PuppyType.Poodle entrance_fee first_place_prize current_savings = 7) ∧
  (min_weeks_to_afford PuppyType.Beagle entrance_fee first_place_prize current_savings = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_weeks_for_puppies_l1153_115363


namespace NUMINAMATH_CALUDE_markup_percentages_correct_l1153_115379

/-- Represents an item with its purchase price, overhead percentage, and desired net profit. -/
structure Item where
  purchase_price : ℕ
  overhead_percent : ℕ
  net_profit : ℕ

/-- Calculates the selling price of an item, rounded up to the nearest whole dollar. -/
def selling_price (item : Item) : ℕ :=
  let total_cost := item.purchase_price + (item.purchase_price * item.overhead_percent / 100) + item.net_profit
  (total_cost + 99) / 100 * 100

/-- Calculates the markup percentage for an item, rounded up to the nearest whole percent. -/
def markup_percentage (item : Item) : ℕ :=
  let markup := selling_price item - item.purchase_price
  ((markup * 100 + item.purchase_price - 1) / item.purchase_price)

theorem markup_percentages_correct (item_a item_b item_c : Item) : 
  item_a.purchase_price = 48 ∧ 
  item_a.overhead_percent = 20 ∧ 
  item_a.net_profit = 12 ∧
  item_b.purchase_price = 36 ∧ 
  item_b.overhead_percent = 15 ∧ 
  item_b.net_profit = 8 ∧
  item_c.purchase_price = 60 ∧ 
  item_c.overhead_percent = 25 ∧ 
  item_c.net_profit = 16 →
  markup_percentage item_a = 46 ∧
  markup_percentage item_b = 39 ∧
  markup_percentage item_c = 52 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentages_correct_l1153_115379


namespace NUMINAMATH_CALUDE_line_invariant_under_transformation_l1153_115373

def transformation (a b : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (-x + a*y, b*x + 3*y)

theorem line_invariant_under_transformation (a b : ℝ) :
  (∀ x y : ℝ, 2*x - y = 3 → 
    let (x', y') := transformation a b x y
    2*x' - y' = 3) →
  a = 1 ∧ b = -4 := by
sorry

end NUMINAMATH_CALUDE_line_invariant_under_transformation_l1153_115373


namespace NUMINAMATH_CALUDE_ava_remaining_distance_l1153_115384

/-- The remaining distance for Ava to finish the race -/
def remaining_distance (race_length : ℕ) (distance_covered : ℕ) : ℕ :=
  race_length - distance_covered

/-- Proof that Ava's remaining distance is 167 meters -/
theorem ava_remaining_distance :
  remaining_distance 1000 833 = 167 := by
  sorry

end NUMINAMATH_CALUDE_ava_remaining_distance_l1153_115384


namespace NUMINAMATH_CALUDE_x_range_for_negative_f_l1153_115354

def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

theorem x_range_for_negative_f :
  (∀ x : ℝ, ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0) →
  (∀ x : ℝ, f (-1) x < 0 ∧ f 1 x < 0 → x ∈ Set.Ioo 1 2) :=
sorry

end NUMINAMATH_CALUDE_x_range_for_negative_f_l1153_115354


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1153_115345

theorem integral_sqrt_one_minus_x_squared_plus_x (f : ℝ → ℝ) :
  (∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1153_115345


namespace NUMINAMATH_CALUDE_current_rabbits_in_cage_l1153_115399

/-- The number of rabbits Jasper saw in the park -/
def rabbits_in_park : ℕ := 60

/-- The number of rabbits currently in the cage -/
def rabbits_in_cage : ℕ := 13

/-- The number of rabbits to be added to the cage -/
def rabbits_to_add : ℕ := 7

theorem current_rabbits_in_cage :
  rabbits_in_cage + rabbits_to_add = rabbits_in_park / 3 ∧
  rabbits_in_cage = 13 :=
by sorry

end NUMINAMATH_CALUDE_current_rabbits_in_cage_l1153_115399


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1153_115352

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9*x = x*y) :
  x + y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ y₀ + 9*x₀ = x₀*y₀ ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1153_115352


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1153_115308

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinguishable balls into 4 distinguishable boxes is 4^5 -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1153_115308


namespace NUMINAMATH_CALUDE_brianna_books_to_reread_l1153_115313

/-- The number of books Brianna reads per month -/
def books_per_month : ℕ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of new books Brianna was given as a gift -/
def books_gifted : ℕ := 6

/-- The number of new books Brianna bought -/
def books_bought : ℕ := 8

/-- The number of new books Brianna plans to borrow from the library -/
def books_borrowed : ℕ := books_bought - 2

/-- The total number of books Brianna needs for the year -/
def total_books_needed : ℕ := books_per_month * months_in_year

/-- The total number of new books Brianna will have -/
def total_new_books : ℕ := books_gifted + books_bought + books_borrowed

/-- The number of old books Brianna needs to reread -/
def old_books_to_reread : ℕ := total_books_needed - total_new_books

theorem brianna_books_to_reread : old_books_to_reread = 4 := by
  sorry

end NUMINAMATH_CALUDE_brianna_books_to_reread_l1153_115313


namespace NUMINAMATH_CALUDE_vector_dot_product_equals_22_l1153_115321

/-- Given two vectors AB and BC in ℝ², where BC has a magnitude of √10,
    prove that the dot product of AB and AC equals 22. -/
theorem vector_dot_product_equals_22 
  (AB : ℝ × ℝ) 
  (BC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : ∃ t > 0, BC = (3, t)) 
  (h3 : Real.sqrt ((BC.1)^2 + (BC.2)^2) = Real.sqrt 10) : 
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_equals_22_l1153_115321


namespace NUMINAMATH_CALUDE_age_ratio_after_years_l1153_115382

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8
def years_elapsed : ℕ := 4

theorem age_ratio_after_years : 
  (suzy_current_age + years_elapsed) / (mary_current_age + years_elapsed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_years_l1153_115382


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1153_115397

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The statement that "a = 0" is a necessary but not sufficient condition for "a + bi to be purely imaginary". -/
theorem a_zero_necessary_not_sufficient (a b : ℝ) :
  (∀ z : ℂ, z = a + b * I → is_purely_imaginary z → a = 0) ∧
  (∃ z : ℂ, z = a + b * I ∧ a = 0 ∧ ¬is_purely_imaginary z) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1153_115397


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1153_115336

theorem unique_solution_for_equation : ∃! (x y : ℕ), 
  x < 10 ∧ y < 10 ∧ (10 + x) * (200 + 10 * y + 7) = 5166 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1153_115336


namespace NUMINAMATH_CALUDE_complex_number_in_quadrant_iv_l1153_115323

/-- The complex number (2-i)/(1+i) corresponds to a point in Quadrant IV of the complex plane -/
theorem complex_number_in_quadrant_iv : 
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_quadrant_iv_l1153_115323


namespace NUMINAMATH_CALUDE_average_score_range_l1153_115387

/-- Represents the score distribution in the math competition --/
structure ScoreDistribution where
  score_100 : ℕ
  score_90_99 : ℕ
  score_80_89 : ℕ
  score_70_79 : ℕ
  score_60_69 : ℕ
  score_50_59 : ℕ
  score_48 : ℕ

/-- Calculates the minimum possible average score --/
def min_average_score (sd : ScoreDistribution) : ℚ :=
  (100 * sd.score_100 + 90 * sd.score_90_99 + 80 * sd.score_80_89 + 70 * sd.score_70_79 +
   60 * sd.score_60_69 + 50 * sd.score_50_59 + 48 * sd.score_48) /
  (sd.score_100 + sd.score_90_99 + sd.score_80_89 + sd.score_70_79 +
   sd.score_60_69 + sd.score_50_59 + sd.score_48)

/-- Calculates the maximum possible average score --/
def max_average_score (sd : ScoreDistribution) : ℚ :=
  (100 * sd.score_100 + 99 * sd.score_90_99 + 89 * sd.score_80_89 + 79 * sd.score_70_79 +
   69 * sd.score_60_69 + 59 * sd.score_50_59 + 48 * sd.score_48) /
  (sd.score_100 + sd.score_90_99 + sd.score_80_89 + sd.score_70_79 +
   sd.score_60_69 + sd.score_50_59 + sd.score_48)

/-- The score distribution for the given problem --/
def zhi_cheng_distribution : ScoreDistribution :=
  { score_100 := 2
  , score_90_99 := 9
  , score_80_89 := 17
  , score_70_79 := 28
  , score_60_69 := 36
  , score_50_59 := 7
  , score_48 := 1
  }

/-- Theorem stating the range of the overall average score --/
theorem average_score_range :
  min_average_score zhi_cheng_distribution ≥ 68.88 ∧
  max_average_score zhi_cheng_distribution ≤ 77.61 := by
  sorry

end NUMINAMATH_CALUDE_average_score_range_l1153_115387


namespace NUMINAMATH_CALUDE_count_strictly_ordered_three_digit_numbers_l1153_115301

/-- The number of three-digit numbers with digits from 1 to 9 in strictly increasing or decreasing order -/
def strictly_ordered_three_digit_numbers : ℕ :=
  2 * (Nat.choose 9 3)

/-- Theorem: The number of three-digit numbers with digits from 1 to 9 
    in strictly increasing or decreasing order is 168 -/
theorem count_strictly_ordered_three_digit_numbers :
  strictly_ordered_three_digit_numbers = 168 := by
  sorry

end NUMINAMATH_CALUDE_count_strictly_ordered_three_digit_numbers_l1153_115301


namespace NUMINAMATH_CALUDE_joao_salary_height_l1153_115394

/-- Conversion rate from real to cruzado -/
def real_to_cruzado : ℝ := 2750000000

/-- João's monthly salary in reais -/
def joao_salary : ℝ := 640

/-- Height of 100 cruzado notes in centimeters -/
def stack_height : ℝ := 1.5

/-- Number of cruzado notes in a stack -/
def notes_per_stack : ℝ := 100

/-- Conversion factor from centimeters to kilometers -/
def cm_to_km : ℝ := 100000

theorem joao_salary_height : 
  (joao_salary * real_to_cruzado / notes_per_stack * stack_height) / cm_to_km = 264000 := by
  sorry

end NUMINAMATH_CALUDE_joao_salary_height_l1153_115394


namespace NUMINAMATH_CALUDE_candy_box_prices_l1153_115349

theorem candy_box_prices (total_money : ℕ) (metal_boxes : ℕ) (paper_boxes : ℕ) :
  -- A buys 4 fewer boxes than B
  metal_boxes + 4 = paper_boxes →
  -- A has 6 yuan left
  ∃ (metal_price : ℕ), metal_price * metal_boxes + 6 = total_money →
  -- B uses all the money
  ∃ (paper_price : ℕ), paper_price * paper_boxes = total_money →
  -- If A used three times his original amount of money
  ∃ (new_metal_boxes : ℕ), 
    -- He would buy 31 more boxes than B
    new_metal_boxes = paper_boxes + 31 →
    -- And still have 6 yuan left
    metal_price * new_metal_boxes + 6 = 3 * total_money →
  -- Then the price of metal boxes is 12 yuan
  metal_price = 12 ∧
  -- And the price of paper boxes is 10 yuan
  paper_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_prices_l1153_115349


namespace NUMINAMATH_CALUDE_green_balls_count_l1153_115310

theorem green_balls_count (blue_count : ℕ) (total_count : ℕ) 
  (h1 : blue_count = 8)
  (h2 : (blue_count : ℚ) / total_count = 1 / 3) :
  total_count - blue_count = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l1153_115310


namespace NUMINAMATH_CALUDE_carpenter_sinks_bought_l1153_115328

/-- The number of sinks needed per house -/
def sinksPerHouse : ℕ := 6

/-- The number of houses that can be covered with the bought sinks -/
def housesCovered : ℕ := 44

/-- The total number of sinks bought by the carpenter -/
def totalSinksBought : ℕ := sinksPerHouse * housesCovered

theorem carpenter_sinks_bought : totalSinksBought = 264 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_sinks_bought_l1153_115328


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l1153_115383

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the quadrants
def first_quadrant (p : Point) : Prop := p.1 > 0 ∧ p.2 > 0
def second_quadrant (p : Point) : Prop := p.1 < 0 ∧ p.2 > 0
def third_quadrant (p : Point) : Prop := p.1 < 0 ∧ p.2 < 0
def fourth_quadrant (p : Point) : Prop := p.1 > 0 ∧ p.2 < 0

-- Define the points P and Q
def P (b : ℝ) : Point := (2, b)
def Q (b : ℝ) : Point := (b, -2)

-- State the theorem
theorem point_quadrant_relation (b : ℝ) :
  fourth_quadrant (P b) → third_quadrant (Q b) :=
by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_relation_l1153_115383


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1153_115309

theorem unique_solution_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1153_115309


namespace NUMINAMATH_CALUDE_betty_payment_l1153_115305

-- Define the given conditions
def doug_age : ℕ := 40
def sum_of_ages : ℕ := 90
def num_packs : ℕ := 20

-- Define Betty's age
def betty_age : ℕ := sum_of_ages - doug_age

-- Define the cost of a pack of nuts
def pack_cost : ℕ := 2 * betty_age

-- Theorem to prove
theorem betty_payment : betty_age * num_packs * 2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_betty_payment_l1153_115305


namespace NUMINAMATH_CALUDE_athletes_meeting_distance_l1153_115314

theorem athletes_meeting_distance (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  (∃ x : ℝ, x > 0 ∧ 
    300 / v₁ = (x - 300) / v₂ ∧ 
    (x + 100) / v₁ = (x - 100) / v₂) → 
  (∃ x : ℝ, x = 500) :=
by sorry

end NUMINAMATH_CALUDE_athletes_meeting_distance_l1153_115314


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1153_115378

/-- The number of ways to arrange n people in a row -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of people to be seated -/
def total_people : ℕ := 8

/-- The number of ways to arrange people with restrictions -/
def seating_arrangements : ℕ :=
  2 * factorial (total_people - 1) - 2 * factorial (total_people - 2) * factorial 2

theorem correct_seating_arrangements :
  seating_arrangements = 7200 := by
  sorry

#eval seating_arrangements

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1153_115378


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l1153_115311

/-- Prove that the number of meters of cloth sold is 85, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_problem (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
  (h1 : total_selling_price = 8925)
  (h2 : profit_per_meter = 35)
  (h3 : cost_price_per_meter = 70) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l1153_115311


namespace NUMINAMATH_CALUDE_new_tv_width_l1153_115346

theorem new_tv_width (first_tv_width : ℝ) (first_tv_height : ℝ) (first_tv_cost : ℝ)
                     (new_tv_height : ℝ) (new_tv_cost : ℝ) :
  first_tv_width = 24 →
  first_tv_height = 16 →
  first_tv_cost = 672 →
  new_tv_height = 32 →
  new_tv_cost = 1152 →
  (first_tv_cost / (first_tv_width * first_tv_height)) =
    (new_tv_cost / (new_tv_height * (new_tv_cost / (new_tv_height * (first_tv_cost / (first_tv_width * first_tv_height) - 1))))) + 1 →
  new_tv_cost / (new_tv_height * (first_tv_cost / (first_tv_width * first_tv_height) - 1)) = 48 :=
by
  sorry

#check new_tv_width

end NUMINAMATH_CALUDE_new_tv_width_l1153_115346


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l1153_115319

theorem wrong_mark_calculation (n : Nat) (initial_avg correct_avg : ℝ) (correct_mark : ℝ) :
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 95 ∧ 
  correct_mark = 10 →
  ∃ wrong_mark : ℝ,
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    wrong_mark = 60 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l1153_115319


namespace NUMINAMATH_CALUDE_inequality_proof_l1153_115344

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1153_115344


namespace NUMINAMATH_CALUDE_largest_valid_selection_l1153_115388

/-- Represents a selection of squares on an n × n grid -/
def Selection (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle contains a selected square -/
def containsSelected (s : Selection n) (x y w h : ℕ) : Prop :=
  ∃ (i j : Fin n), s i j ∧ x ≤ i.val ∧ i.val < x + w ∧ y ≤ j.val ∧ j.val < y + h

/-- Checks if a selection satisfies the condition for all rectangles -/
def validSelection (n : ℕ) (s : Selection n) : Prop :=
  ∀ (x y w h : ℕ), x + w ≤ n → y + h ≤ n → w * h ≥ n → containsSelected s x y w h

/-- The main theorem stating that 7 is the largest n satisfying the condition -/
theorem largest_valid_selection :
  (∀ n : ℕ, n ≤ 7 → ∃ s : Selection n, (∀ i : Fin n, ∃! j : Fin n, s i j) ∧ validSelection n s) ∧
  (∀ n : ℕ, n > 7 → ¬∃ s : Selection n, (∀ i : Fin n, ∃! j : Fin n, s i j) ∧ validSelection n s) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_selection_l1153_115388


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1153_115326

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 2) : 
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 27/8 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 2 ∧
    1 / (2*a₀ + b₀) + 1 / (2*b₀ + c₀) + 1 / (2*c₀ + a₀) = 27/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1153_115326


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1153_115386

theorem sum_of_two_numbers (s l : ℕ) : s = 9 → l = 4 * s → s + l = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1153_115386


namespace NUMINAMATH_CALUDE_no_solution_l1153_115392

/-- Product of digits of a natural number in base ten -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem: no natural number satisfies the given equation -/
theorem no_solution :
  ∀ x : ℕ, productOfDigits x ≠ x^2 - 10*x - 22 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1153_115392


namespace NUMINAMATH_CALUDE_intersection_A_B_l1153_115355

def A : Set ℝ := {-3, -1, 1, 2}
def B : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1153_115355


namespace NUMINAMATH_CALUDE_april_rainfall_calculation_l1153_115317

def march_rainfall : ℝ := 0.81
def april_decrease : ℝ := 0.35

theorem april_rainfall_calculation :
  march_rainfall - april_decrease = 0.46 := by sorry

end NUMINAMATH_CALUDE_april_rainfall_calculation_l1153_115317


namespace NUMINAMATH_CALUDE_decimal_division_l1153_115366

theorem decimal_division : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l1153_115366


namespace NUMINAMATH_CALUDE_arrangement_count_l1153_115342

/-- The number of volunteers -/
def num_volunteers : ℕ := 5

/-- The number of elderly people -/
def num_elderly : ℕ := 2

/-- The number of positions where the elderly pair can be placed -/
def elderly_pair_positions : ℕ := num_volunteers - 1

/-- The number of arrangements of volunteers -/
def volunteer_arrangements : ℕ := Nat.factorial num_volunteers

/-- The number of arrangements of elderly people -/
def elderly_arrangements : ℕ := Nat.factorial num_elderly

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := elderly_pair_positions * volunteer_arrangements * elderly_arrangements

theorem arrangement_count : total_arrangements = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1153_115342


namespace NUMINAMATH_CALUDE_journey_time_calculation_l1153_115389

/-- Given a constant speed, if a 200-mile journey takes 5 hours, 
    then a 120-mile journey will take 3 hours. -/
theorem journey_time_calculation (speed : ℝ) 
  (h1 : speed > 0)
  (h2 : 200 = speed * 5) : 
  120 = speed * 3 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l1153_115389


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1153_115357

theorem triangle_angle_calculation (a b : ℝ) (B : ℝ) (hA : a = Real.sqrt 2) (hB : B = 45 * π / 180) (hb : b = 2) :
  ∃ (A : ℝ), A = 30 * π / 180 ∧ a / Real.sin A = b / Real.sin B := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1153_115357


namespace NUMINAMATH_CALUDE_binomial_and_permutation_l1153_115304

theorem binomial_and_permutation :
  (Nat.choose 8 5 = 56) ∧
  (Nat.factorial 5 / Nat.factorial 2 = 60) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_permutation_l1153_115304


namespace NUMINAMATH_CALUDE_domain_of_composed_function_l1153_115325

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_composed_function :
  (∀ x ∈ Set.Icc 0 3, f (x + 1) ∈ Set.range f) →
  {x : ℝ | f (2^x) ∈ Set.range f} = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_domain_of_composed_function_l1153_115325


namespace NUMINAMATH_CALUDE_function_has_positive_zero_l1153_115324

/-- The function f(x) = xe^x - ax - 1 has at least one positive zero for any real a. -/
theorem function_has_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ x * Real.exp x - a * x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_has_positive_zero_l1153_115324


namespace NUMINAMATH_CALUDE_landscape_ratio_l1153_115337

theorem landscape_ratio (length : ℝ) (playground_area : ℝ) (playground_ratio : ℝ) :
  length = 240 →
  playground_area = 1200 →
  playground_ratio = 1 / 6 →
  ∃ breadth : ℝ, breadth > 0 ∧ length / breadth = 8 := by
  sorry

end NUMINAMATH_CALUDE_landscape_ratio_l1153_115337


namespace NUMINAMATH_CALUDE_problem_solution_l1153_115302

theorem problem_solution (x y : ℝ) (h1 : y = 3) (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1153_115302


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1153_115322

theorem negative_fraction_comparison :
  -((4 : ℚ) / 5) < -((3 : ℚ) / 4) := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1153_115322


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1153_115365

theorem polynomial_remainder (x : ℤ) : (x^2008 + 2008*x + 2008) % (x + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1153_115365


namespace NUMINAMATH_CALUDE_not_perfect_square_sum_l1153_115358

theorem not_perfect_square_sum (x y : ℤ) : 
  ∃ (n : ℤ), (x^2 + x + 1)^2 + (y^2 + y + 1)^2 ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_sum_l1153_115358


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1153_115312

/-- Given two points M and N that are symmetric with respect to the x-axis,
    prove that the sum of their x and y coordinates is -3. -/
theorem symmetric_points_sum (b a : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (-2, b) ∧ 
    N = (a, 1) ∧ 
    (M.1 = N.1 ∧ M.2 = -N.2)) → 
  a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1153_115312


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l1153_115329

theorem unique_root_quadratic (b c : ℝ) : 
  (∃! x : ℝ, x^2 + b*x + c = 0) → 
  (b = c + 1) → 
  c = 1 :=
by
  sorry

#check unique_root_quadratic

end NUMINAMATH_CALUDE_unique_root_quadratic_l1153_115329


namespace NUMINAMATH_CALUDE_smallest_equal_purchase_l1153_115369

theorem smallest_equal_purchase (nuts : Nat) (bolts : Nat) (washers : Nat)
  (h_nuts : nuts = 13)
  (h_bolts : bolts = 8)
  (h_washers : washers = 17) :
  Nat.lcm (Nat.lcm nuts bolts) washers = 1768 := by
  sorry

end NUMINAMATH_CALUDE_smallest_equal_purchase_l1153_115369
