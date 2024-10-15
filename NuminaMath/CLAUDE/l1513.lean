import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_x5_expansion_l1513_151371

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^5 in the expansion of (1-x^3)(1+x)^10
def coefficient_x5 : ℕ := binomial 10 5 - binomial 10 2

-- Theorem statement
theorem coefficient_x5_expansion :
  coefficient_x5 = 207 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5_expansion_l1513_151371


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1513_151373

/-- Given a parabola y = -ax²/4 + ax + c and three points on it, 
    prove that if y₁ > y₃ ≥ y₂ and y₂ is the vertex, then |x₁ - x₂| > |x₃ - x₂| -/
theorem parabola_point_distance (a c x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  y₁ = -a * x₁^2 / 4 + a * x₁ + c →
  y₂ = -a * x₂^2 / 4 + a * x₂ + c →
  y₃ = -a * x₃^2 / 4 + a * x₃ + c →
  y₂ = a + c →
  y₁ > y₃ →
  y₃ ≥ y₂ →
  |x₁ - x₂| > |x₃ - x₂| := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1513_151373


namespace NUMINAMATH_CALUDE_fifth_degree_monomial_n_value_l1513_151372

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree (n : ℕ) : ℕ := n + 2 + 1

/-- A monomial 4a^nb^2c is a fifth-degree monomial if its degree is 5 -/
def is_fifth_degree (n : ℕ) : Prop := degree n = 5

theorem fifth_degree_monomial_n_value :
  ∀ n : ℕ, is_fifth_degree n → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fifth_degree_monomial_n_value_l1513_151372


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l1513_151352

def U : Finset ℕ := {1,2,3,4,5,6}
def M : Finset ℕ := {2,3,5}
def N : Finset ℕ := {4,5}

theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l1513_151352


namespace NUMINAMATH_CALUDE_constant_term_of_x_minus_inverse_x_power_8_l1513_151379

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the constant term of (x - 1/x)^n
def constantTerm (n : ℕ) : ℤ :=
  if n % 2 = 0
  then (-1)^(n/2) * binomial n (n/2)
  else 0

-- Theorem statement
theorem constant_term_of_x_minus_inverse_x_power_8 :
  constantTerm 8 = 70 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_x_minus_inverse_x_power_8_l1513_151379


namespace NUMINAMATH_CALUDE_no_prime_pair_divisibility_l1513_151321

theorem no_prime_pair_divisibility : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ (p * q ∣ (2^p - 1) * (2^q - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pair_divisibility_l1513_151321


namespace NUMINAMATH_CALUDE_ajay_ride_distance_l1513_151322

/-- Given Ajay's speed and travel time, calculate the distance he rides -/
theorem ajay_ride_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 50 → time = 30 → distance = speed * time → distance = 1500 :=
by
  sorry

#check ajay_ride_distance

end NUMINAMATH_CALUDE_ajay_ride_distance_l1513_151322


namespace NUMINAMATH_CALUDE_portion_filled_in_twenty_minutes_l1513_151381

/-- Represents the portion of a cistern filled by a pipe in a given time. -/
def portion_filled (time : ℝ) : ℝ := sorry

/-- The time it takes to fill a certain portion of the cistern. -/
def fill_time : ℝ := 20

/-- Theorem stating that the portion filled in 20 minutes is 1. -/
theorem portion_filled_in_twenty_minutes :
  portion_filled fill_time = 1 := by sorry

end NUMINAMATH_CALUDE_portion_filled_in_twenty_minutes_l1513_151381


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1513_151390

theorem diophantine_equation_solution (x y : ℤ) :
  5 * x - 7 * y = 3 →
  ∃ t : ℤ, x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1513_151390


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l1513_151342

theorem quadratic_always_negative (m k : ℝ) :
  (∀ x : ℝ, x^2 - m*x - k + m < 0) ↔ k > m - m^2/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l1513_151342


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1513_151377

/-- A rectangle with perimeter 60 meters and area 224 square meters has a longer side of 16 meters. -/
theorem rectangle_longer_side (x y : ℝ) (h_perimeter : x + y = 30) (h_area : x * y = 224) 
  (h_x_longer : x ≥ y) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1513_151377


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1513_151358

theorem quadratic_complete_square (r s : ℚ) : 
  (∀ x, 7 * x^2 - 21 * x - 56 = 0 ↔ (x + r)^2 = s) → 
  r + s = 35/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1513_151358


namespace NUMINAMATH_CALUDE_complete_square_sum_l1513_151354

theorem complete_square_sum (a b c : ℝ) (r s : ℝ) :
  (6 * a^2 - 30 * a - 36 = 0) →
  ((a + r)^2 = s) →
  (6 * a^2 - 30 * a - 36 = 6 * ((a + r)^2 - s)) →
  (r + s = 9.75) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1513_151354


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l1513_151350

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory with a conveyor belt and sampling process -/
structure Factory where
  sampleInterval : ℕ  -- Time interval between samples in minutes
  sampleLocation : String  -- Description of the sample location

/-- Determines the sampling method based on the factory's sampling process -/
def determineSamplingMethod (f : Factory) : SamplingMethod :=
  sorry

/-- Theorem stating that the described sampling method is systematic sampling -/
theorem factory_sampling_is_systematic (f : Factory) 
  (h1 : f.sampleInterval = 10)
  (h2 : f.sampleLocation = "specific location on the conveyor belt") :
  determineSamplingMethod f = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l1513_151350


namespace NUMINAMATH_CALUDE_probability_different_suits_l1513_151347

def deck_size : ℕ := 60
def num_suits : ℕ := 4
def cards_per_suit : ℕ := 15

theorem probability_different_suits :
  let prob_diff_suits := (deck_size - cards_per_suit) / (deck_size * (deck_size - 1))
  prob_diff_suits = 45 / 236 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_suits_l1513_151347


namespace NUMINAMATH_CALUDE_village_walk_speeds_l1513_151383

/-- Proves that given the conditions of the problem, the speeds of the two people are 2 km/h and 5 km/h respectively. -/
theorem village_walk_speeds (distance : ℝ) (speed_diff : ℝ) (time_diff : ℝ)
  (h1 : distance = 10)
  (h2 : speed_diff = 3)
  (h3 : time_diff = 3)
  (h4 : ∀ x : ℝ, distance / x = distance / (x + speed_diff) + time_diff → x = 2) :
  ∃ (speed1 speed2 : ℝ), speed1 = 2 ∧ speed2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_village_walk_speeds_l1513_151383


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1513_151300

theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![(-2), 1]
  let b : Fin 2 → ℝ := ![x, 2]
  (∀ i : Fin 2, a i * b i = 0) →
  x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1513_151300


namespace NUMINAMATH_CALUDE_product_sum_relation_l1513_151394

theorem product_sum_relation (a b c : ℚ) 
  (h1 : a * b * c = 2 * (a + b + c) + 14)
  (h2 : b = 8)
  (h3 : c = 5) :
  (c - a)^2 + b = 8513 / 361 := by sorry

end NUMINAMATH_CALUDE_product_sum_relation_l1513_151394


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1513_151333

theorem slope_angle_of_line (x y : ℝ) :
  y = -Real.sqrt 3 * x + 1 → Real.arctan (-Real.sqrt 3) * (180 / Real.pi) = 120 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1513_151333


namespace NUMINAMATH_CALUDE_digit_123_is_1_l1513_151370

/-- The decimal representation of 47/740 -/
def decimal_rep : ℚ := 47 / 740

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 12

/-- The position we're interested in -/
def target_position : ℕ := 123

/-- The function that returns the nth digit after the decimal point in the decimal representation of 47/740 -/
noncomputable def nth_digit (n : ℕ) : ℕ :=
  sorry

theorem digit_123_is_1 : nth_digit target_position = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_123_is_1_l1513_151370


namespace NUMINAMATH_CALUDE_omicron_ba3_sample_size_l1513_151301

/-- The number of Omicron BA.3 virus strains in a stratified random sample -/
theorem omicron_ba3_sample_size 
  (total_strains : ℕ) 
  (ba3_strains : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_strains = 120) 
  (h2 : ba3_strains = 40) 
  (h3 : sample_size = 30) :
  (ba3_strains : ℚ) / total_strains * sample_size = 10 :=
sorry

end NUMINAMATH_CALUDE_omicron_ba3_sample_size_l1513_151301


namespace NUMINAMATH_CALUDE_jim_siblings_l1513_151356

-- Define the characteristics
inductive EyeColor
| Blue
| Brown

inductive HairColor
| Blond
| Black

inductive GlassesWorn
| Yes
| No

-- Define a student structure
structure Student where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  glassesWorn : GlassesWorn

-- Define the list of students
def students : List Student := [
  ⟨"Benjamin", EyeColor.Blue, HairColor.Blond, GlassesWorn.Yes⟩,
  ⟨"Jim", EyeColor.Brown, HairColor.Blond, GlassesWorn.No⟩,
  ⟨"Nadeen", EyeColor.Brown, HairColor.Black, GlassesWorn.Yes⟩,
  ⟨"Austin", EyeColor.Blue, HairColor.Black, GlassesWorn.No⟩,
  ⟨"Tevyn", EyeColor.Blue, HairColor.Blond, GlassesWorn.Yes⟩,
  ⟨"Sue", EyeColor.Brown, HairColor.Blond, GlassesWorn.No⟩
]

-- Define a function to check if two students share at least one characteristic
def shareCharacteristic (s1 s2 : Student) : Prop :=
  s1.eyeColor = s2.eyeColor ∨ s1.hairColor = s2.hairColor ∨ s1.glassesWorn = s2.glassesWorn

-- Define a function to check if three students are siblings
def areSiblings (s1 s2 s3 : Student) : Prop :=
  shareCharacteristic s1 s2 ∧ shareCharacteristic s2 s3 ∧ shareCharacteristic s1 s3

-- Theorem statement
theorem jim_siblings :
  ∃ (jim sue benjamin : Student),
    jim ∈ students ∧ sue ∈ students ∧ benjamin ∈ students ∧
    jim.name = "Jim" ∧ sue.name = "Sue" ∧ benjamin.name = "Benjamin" ∧
    areSiblings jim sue benjamin ∧
    (∀ (other : Student), other ∈ students → other.name ≠ "Jim" → other.name ≠ "Sue" → other.name ≠ "Benjamin" →
      ¬(areSiblings jim sue other ∨ areSiblings jim benjamin other ∨ areSiblings sue benjamin other)) :=
sorry

end NUMINAMATH_CALUDE_jim_siblings_l1513_151356


namespace NUMINAMATH_CALUDE_max_correct_answers_l1513_151334

/-- Represents an exam with a specific scoring system and result. -/
structure Exam where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Represents a possible breakdown of answers in an exam. -/
structure ExamResult where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ

/-- Checks if an ExamResult is valid for a given Exam. -/
def is_valid_result (e : Exam) (r : ExamResult) : Prop :=
  r.correct + r.incorrect + r.unanswered = e.total_questions ∧
  r.correct * e.correct_points + r.incorrect * e.incorrect_points = e.total_score

/-- Theorem: The maximum number of correct answers for the given exam is 33. -/
theorem max_correct_answers (e : Exam) :
  e.total_questions = 60 ∧ e.correct_points = 5 ∧ e.incorrect_points = -1 ∧ e.total_score = 140 →
  (∃ (r : ExamResult), is_valid_result e r ∧
    ∀ (r' : ExamResult), is_valid_result e r' → r'.correct ≤ r.correct) ∧
  (∃ (r : ExamResult), is_valid_result e r ∧ r.correct = 33) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1513_151334


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1513_151395

/-- The quadratic function f(x) = -3x^2 - 2 has its vertex at (0, -2). -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x => -3 * x^2 - 2
  (∀ x, f x ≤ f 0) ∧ f 0 = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1513_151395


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1513_151361

theorem cement_mixture_weight :
  ∀ (W : ℚ),
  (2 / 7 : ℚ) * W +  -- Sand
  (3 / 7 : ℚ) * W +  -- Water
  (1 / 14 : ℚ) * W + -- Gravel
  (1 / 14 : ℚ) * W + -- Cement
  12 = W             -- Crushed stones
  →
  W = 84 :=
by sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l1513_151361


namespace NUMINAMATH_CALUDE_abs_negative_2023_l1513_151307

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l1513_151307


namespace NUMINAMATH_CALUDE_angle_ABD_measure_l1513_151323

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles in the quadrilateral
def angle_ABC (q : Quadrilateral) : ℝ := 120
def angle_DAB (q : Quadrilateral) : ℝ := 30
def angle_ADB (q : Quadrilateral) : ℝ := 28

-- Define the theorem
theorem angle_ABD_measure (q : Quadrilateral) :
  angle_ABC q = 120 ∧ angle_DAB q = 30 ∧ angle_ADB q = 28 →
  ∃ (angle_ABD : ℝ), angle_ABD = 122 :=
sorry

end NUMINAMATH_CALUDE_angle_ABD_measure_l1513_151323


namespace NUMINAMATH_CALUDE_f_properties_l1513_151359

-- Define the function f(x) = x³ - 3x² - 9x
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- Theorem statement
theorem f_properties :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ domain → f y ≤ f x) ∧
  (∀ (m : ℝ), ∃ (x : ℝ), x ∈ domain ∧ f x < m) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1513_151359


namespace NUMINAMATH_CALUDE_false_proposition_l1513_151399

-- Define the lines
def line1 : ℝ → ℝ → Prop := λ x y => 6*x + 2*y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y => y = 5 - 3*x
def line3 : ℝ → ℝ → Prop := λ x y => 2*x + 6*y - 4 = 0

-- Define the propositions
def p : Prop := ∀ x y, line1 x y ↔ line2 x y
def q : Prop := ∀ x y, line1 x y → line3 x y

-- Theorem statement
theorem false_proposition : ¬((¬p) ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_false_proposition_l1513_151399


namespace NUMINAMATH_CALUDE_min_value_a_plus_8b_l1513_151332

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∀ x y, x > 0 → y > 0 → 2 * x * y = x + 2 * y → x + 8 * y ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_8b_l1513_151332


namespace NUMINAMATH_CALUDE_pq_satisfies_stewarts_theorem_l1513_151375

/-- Triangle DEF with given side lengths and points P and Q -/
structure TriangleDEF where
  -- Side lengths
  DE : ℝ
  EF : ℝ
  DF : ℝ
  -- P is the midpoint of DE
  P : ℝ × ℝ
  -- Q is the foot of the perpendicular from D to EF
  Q : ℝ × ℝ
  -- Conditions
  de_length : DE = 17
  ef_length : EF = 18
  df_length : DF = 19
  p_midpoint : P.1 = DE / 2
  q_perpendicular : sorry -- This would require more geometric setup

/-- The length of PQ satisfies Stewart's Theorem -/
theorem pq_satisfies_stewarts_theorem (t : TriangleDEF) : 
  ∃ (PQ : ℝ), t.DE * (t.DE / 2)^2 + t.DE * PQ^2 = t.DF * (t.DE / 2) * t.DF + t.EF * (t.DE / 2) * t.EF := by
  sorry

#check pq_satisfies_stewarts_theorem

end NUMINAMATH_CALUDE_pq_satisfies_stewarts_theorem_l1513_151375


namespace NUMINAMATH_CALUDE_union_eq_P_l1513_151376

def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x > 1 ∨ x < -1}

theorem union_eq_P : M ∪ P = P := by
  sorry

end NUMINAMATH_CALUDE_union_eq_P_l1513_151376


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_line_passes_through_fixed_point_l1513_151341

/-- A circle that passes through (1, 0) and is tangent to x = -1 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_F : (center.1 - 1)^2 + center.2^2 = radius^2
  tangent_to_l : center.1 + radius = 1

/-- The trajectory of the center of the TangentCircle -/
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Two distinct points on the trajectory, neither being the origin -/
structure TrajectoryPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  A_on_trajectory : A ∈ trajectory
  B_on_trajectory : B ∈ trajectory
  A_not_origin : A ≠ (0, 0)
  B_not_origin : B ≠ (0, 0)
  A_ne_B : A ≠ B
  y_product_ne_neg16 : A.2 * B.2 ≠ -16

theorem trajectory_is_parabola (c : TangentCircle) : c.center ∈ trajectory := by sorry

theorem line_passes_through_fixed_point (p : TrajectoryPoints) :
  ∃ t : ℝ, t * (p.B.1 - p.A.1) + p.A.1 = 4 ∧ t * (p.B.2 - p.A.2) + p.A.2 = 0 := by sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_line_passes_through_fixed_point_l1513_151341


namespace NUMINAMATH_CALUDE_seryozha_sandwich_candy_impossibility_l1513_151397

theorem seryozha_sandwich_candy_impossibility :
  ¬ ∃ (x y z : ℕ), x + 2*y + 3*z = 100 ∧ 3*x + 4*y + 5*z = 166 :=
by sorry

end NUMINAMATH_CALUDE_seryozha_sandwich_candy_impossibility_l1513_151397


namespace NUMINAMATH_CALUDE_sixteen_is_counterexample_l1513_151313

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_counterexample (n : Nat) : Prop :=
  ¬(is_prime n) ∧ (is_prime (n - 2) ∨ is_prime (n + 2))

theorem sixteen_is_counterexample : is_counterexample 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_is_counterexample_l1513_151313


namespace NUMINAMATH_CALUDE_log_equation_sum_l1513_151338

theorem log_equation_sum (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 5 / Real.log 200) + (B : ℝ) * (Real.log 2 / Real.log 200) = C →
  A + B + C = 6 := by
sorry

end NUMINAMATH_CALUDE_log_equation_sum_l1513_151338


namespace NUMINAMATH_CALUDE_tims_golf_balls_l1513_151344

-- Define the number of dozens Tim has
def tims_dozens : ℕ := 13

-- Define the number of items in a dozen
def items_per_dozen : ℕ := 12

-- Theorem to prove
theorem tims_golf_balls : tims_dozens * items_per_dozen = 156 := by
  sorry

end NUMINAMATH_CALUDE_tims_golf_balls_l1513_151344


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1513_151369

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1513_151369


namespace NUMINAMATH_CALUDE_last_two_digits_of_fraction_l1513_151353

theorem last_two_digits_of_fraction (n : ℕ) : n = 50 * 52 * 54 * 56 * 58 * 60 →
  n / 8000 ≡ 22 [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_fraction_l1513_151353


namespace NUMINAMATH_CALUDE_circle_area_difference_radius_l1513_151349

theorem circle_area_difference_radius 
  (r₁ : ℝ) (r₂ : ℝ) (r₃ : ℝ) 
  (h₁ : r₁ = 21) (h₂ : r₂ = 31) 
  (h₃ : π * r₃^2 = π * r₂^2 - π * r₁^2) : 
  r₃ = 2 * Real.sqrt 130 := by
sorry

end NUMINAMATH_CALUDE_circle_area_difference_radius_l1513_151349


namespace NUMINAMATH_CALUDE_solution_x_chemical_b_percentage_l1513_151327

/-- Represents the composition of a chemical solution -/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions -/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

/-- Given conditions of the problem -/
def problem_conditions : Prop :=
  ∃ (x y : Solution) (mix : Mixture),
    x.a = 0.40 ∧
    y.a = 0.50 ∧
    y.b = 0.50 ∧
    x.a + x.b = 1 ∧
    y.a + y.b = 1 ∧
    mix.x = x ∧
    mix.y = y ∧
    mix.x_ratio = 0.30 ∧
    mix.x_ratio * x.a + (1 - mix.x_ratio) * y.a = 0.47

/-- Theorem statement -/
theorem solution_x_chemical_b_percentage :
  problem_conditions →
  ∃ (x : Solution), x.b = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_chemical_b_percentage_l1513_151327


namespace NUMINAMATH_CALUDE_R_symmetry_l1513_151329

/-- Recursive definition of R_n sequences -/
def R : ℕ → List ℕ
  | 0 => [1]
  | n + 1 =>
    let prev := R n
    List.join (prev.map (fun x => List.range x)) ++ [n + 1]

/-- Main theorem -/
theorem R_symmetry (n : ℕ) (k : ℕ) (h : n > 1) :
  (R n).nthLe k (by sorry) = 1 ↔
  (R n).nthLe ((R n).length - 1 - k) (by sorry) ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_R_symmetry_l1513_151329


namespace NUMINAMATH_CALUDE_boosters_club_average_sales_l1513_151325

/-- Calculates the average monthly sales for the Boosters Club --/
theorem boosters_club_average_sales
  (sales : List ℝ)
  (refund : ℝ)
  (h1 : sales = [90, 75, 55, 130, 110, 85])
  (h2 : refund = 25)
  (h3 : sales.length = 6) :
  (sales.sum - refund) / sales.length = 86.67 := by
  sorry

end NUMINAMATH_CALUDE_boosters_club_average_sales_l1513_151325


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1513_151345

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1513_151345


namespace NUMINAMATH_CALUDE_solve_video_game_problem_l1513_151304

def video_game_problem (total_games : ℕ) (potential_earnings : ℕ) (price_per_game : ℕ) : Prop :=
  let working_games := potential_earnings / price_per_game
  let non_working_games := total_games - working_games
  non_working_games = 8

theorem solve_video_game_problem :
  video_game_problem 16 56 7 :=
sorry

end NUMINAMATH_CALUDE_solve_video_game_problem_l1513_151304


namespace NUMINAMATH_CALUDE_smallest_constant_term_l1513_151384

theorem smallest_constant_term (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -2 ∨ x = 5 ∨ x = 9 ∨ x = -1/2) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ 
      x = -2 ∨ x = 5 ∨ x = 9 ∨ x = -1/2) → 
    e ≤ e') →
  e = 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_term_l1513_151384


namespace NUMINAMATH_CALUDE_rectangular_region_ratio_l1513_151328

theorem rectangular_region_ratio (L W : ℝ) (k : ℝ) : 
  L > 0 → W > 0 → k > 0 →
  L = k * W →
  L * W = 200 →
  2 * W + L = 40 →
  L / W = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_region_ratio_l1513_151328


namespace NUMINAMATH_CALUDE_simplify_fraction_l1513_151360

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (75 * b^3) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1513_151360


namespace NUMINAMATH_CALUDE_lock_problem_l1513_151357

/-- The number of buttons on the lock -/
def num_buttons : ℕ := 10

/-- The number of buttons that need to be pressed simultaneously -/
def buttons_to_press : ℕ := 3

/-- The time taken for each attempt in seconds -/
def time_per_attempt : ℕ := 2

/-- The total number of possible combinations -/
def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

/-- The maximum time needed to try all combinations in seconds -/
def max_time : ℕ := total_combinations * time_per_attempt

/-- The average number of attempts needed -/
def avg_attempts : ℚ := (1 + total_combinations) / 2

/-- The average time needed to open the door in seconds -/
def avg_time : ℚ := avg_attempts * time_per_attempt

/-- The maximum number of attempts possible in 60 seconds -/
def max_attempts_in_minute : ℕ := 60 / time_per_attempt

/-- The probability of opening the door in less than 60 seconds -/
def prob_less_than_minute : ℚ := (max_attempts_in_minute - 1) / total_combinations

theorem lock_problem :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (prob_less_than_minute = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_lock_problem_l1513_151357


namespace NUMINAMATH_CALUDE_largest_n_for_positive_sum_l1513_151365

/-- Given an arithmetic sequence {a_n} where a_1 = 9 and a_5 = 1,
    the largest natural number n for which the sum of the first n terms (S_n) is positive is 9. -/
theorem largest_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 9 →
  a 5 = 1 →
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- sum formula for arithmetic sequence
  (∀ m : ℕ, m > 9 → S m ≤ 0) ∧ S 9 > 0 := by
sorry


end NUMINAMATH_CALUDE_largest_n_for_positive_sum_l1513_151365


namespace NUMINAMATH_CALUDE_angle_at_larger_base_l1513_151378

/-- An isosceles trapezoid with a regular triangle on its smaller base -/
structure IsoscelesTrapezoidWithTriangle where
  /-- The smaller base of the trapezoid -/
  smallerBase : ℝ
  /-- The larger base of the trapezoid -/
  largerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The angle at the larger base of the trapezoid -/
  angle : ℝ
  /-- The area of the trapezoid -/
  areaT : ℝ
  /-- The area of the triangle -/
  areaTriangle : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The triangle is regular -/
  isRegular : True
  /-- The height of the triangle equals the height of the trapezoid -/
  heightEqual : True
  /-- The area of the triangle is 5 times less than the area of the trapezoid -/
  areaRelation : areaT = 5 * areaTriangle

/-- The theorem to be proved -/
theorem angle_at_larger_base (t : IsoscelesTrapezoidWithTriangle) :
  t.angle = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_at_larger_base_l1513_151378


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1513_151387

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : failed_both = 28) :
  100 - (failed_hindi + failed_english - failed_both) = 56 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1513_151387


namespace NUMINAMATH_CALUDE_simplify_expression_l1513_151364

theorem simplify_expression : 2 - (2 / (2 + Real.sqrt 5)) - (2 / (2 - Real.sqrt 5)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1513_151364


namespace NUMINAMATH_CALUDE_days_to_fill_tank_l1513_151319

def tank_capacity : ℝ := 350000 -- in milliliters
def min_daily_collection : ℝ := 1200 -- in milliliters
def max_daily_collection : ℝ := 2100 -- in milliliters

theorem days_to_fill_tank : 
  ∃ (days : ℕ), days = 213 ∧ 
  (tank_capacity / min_daily_collection ≤ days) ∧
  (tank_capacity / max_daily_collection ≤ days) ∧
  (∀ d : ℕ, d < days → d * max_daily_collection < tank_capacity) :=
sorry

end NUMINAMATH_CALUDE_days_to_fill_tank_l1513_151319


namespace NUMINAMATH_CALUDE_A_union_B_eq_l1513_151310

def A : Set ℝ := {x | x^2 - x - 2 < 0}

def B : Set ℝ := {x | x^2 - 3*x < 0}

theorem A_union_B_eq : A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_union_B_eq_l1513_151310


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1513_151314

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x, 4)
  vector_parallel a b → x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1513_151314


namespace NUMINAMATH_CALUDE_red_stripes_on_fifty_flags_l1513_151339

/-- Calculates the total number of red stripes on multiple flags -/
def total_red_stripes (stripes_per_flag : ℕ) (num_flags : ℕ) : ℕ :=
  let remaining_stripes := stripes_per_flag - 1
  let red_remaining := remaining_stripes / 2
  let red_per_flag := red_remaining + 1
  red_per_flag * num_flags

/-- Theorem stating the total number of red stripes on 50 flags -/
theorem red_stripes_on_fifty_flags :
  total_red_stripes 25 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_red_stripes_on_fifty_flags_l1513_151339


namespace NUMINAMATH_CALUDE_optimal_transport_solution_l1513_151346

/-- Represents the optimal solution for transporting cargo -/
structure CargoTransport where
  large_trucks : ℕ
  small_trucks : ℕ
  total_fuel : ℕ

/-- Finds the optimal cargo transport solution -/
def find_optimal_transport (total_cargo : ℕ) (large_capacity : ℕ) (small_capacity : ℕ)
  (large_fuel : ℕ) (small_fuel : ℕ) : CargoTransport :=
  sorry

/-- Theorem stating the optimal solution for the given problem -/
theorem optimal_transport_solution :
  let total_cargo : ℕ := 89
  let large_capacity : ℕ := 7
  let small_capacity : ℕ := 4
  let large_fuel : ℕ := 14
  let small_fuel : ℕ := 9
  let solution := find_optimal_transport total_cargo large_capacity small_capacity large_fuel small_fuel
  solution.total_fuel = 181 ∧
  solution.large_trucks * large_capacity + solution.small_trucks * small_capacity ≥ total_cargo :=
by sorry

end NUMINAMATH_CALUDE_optimal_transport_solution_l1513_151346


namespace NUMINAMATH_CALUDE_equation_solutions_l1513_151366

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^3 + x^2*y + x*y^2 + y^3 = 8*(x^2 + x*y + y^2 + 1)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(8, -2), (-2, 8), (4 + Real.sqrt 15, 4 - Real.sqrt 15), (4 - Real.sqrt 15, 4 + Real.sqrt 15)}

-- Theorem statement
theorem equation_solutions :
  ∀ x y : ℝ, equation x y ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1513_151366


namespace NUMINAMATH_CALUDE_best_overall_value_l1513_151308

structure Box where
  brand : String
  size : Nat
  price : Rat
  quality : Rat

def pricePerOunce (b : Box) : Rat :=
  b.price / b.size

def overallValue (b : Box) : Rat :=
  b.quality / (pricePerOunce b)

theorem best_overall_value (box1 box2 box3 box4 : Box) 
  (h1 : box1 = { brand := "A", size := 30, price := 480/100, quality := 9/2 })
  (h2 : box2 = { brand := "A", size := 20, price := 340/100, quality := 9/2 })
  (h3 : box3 = { brand := "B", size := 15, price := 200/100, quality := 39/10 })
  (h4 : box4 = { brand := "B", size := 25, price := 325/100, quality := 39/10 }) :
  overallValue box1 ≥ overallValue box2 ∧ 
  overallValue box1 ≥ overallValue box3 ∧ 
  overallValue box1 ≥ overallValue box4 := by
  sorry

#check best_overall_value

end NUMINAMATH_CALUDE_best_overall_value_l1513_151308


namespace NUMINAMATH_CALUDE_product_inequality_l1513_151362

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1513_151362


namespace NUMINAMATH_CALUDE_inequality_proof_l1513_151368

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1513_151368


namespace NUMINAMATH_CALUDE_calculation_proof_l1513_151318

theorem calculation_proof :
  ((-1/2) * (-8) + (-6) = -2) ∧
  (-1^4 - 2 / (-1/3) - |-9| = -4) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1513_151318


namespace NUMINAMATH_CALUDE_range_of_a_is_closed_interval_two_three_l1513_151385

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

def g (a x : ℝ) : ℝ := x^2 - a*x - a + 3

theorem range_of_a_is_closed_interval_two_three :
  ∃ (a : ℝ), ∀ x₁ x₂ : ℝ,
    f x₁ = 0 ∧ g a x₂ = 0 ∧ |x₁ - x₂| ≤ 1 →
    a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_is_closed_interval_two_three_l1513_151385


namespace NUMINAMATH_CALUDE_eccentricity_is_sqrt_five_l1513_151302

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- Represents a point on a hyperbola -/
structure PointOnHyperbola {a b : ℝ} (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The left and right foci of a hyperbola -/
def foci {a b : ℝ} (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The distance between a point and a focus -/
def dist_to_focus {a b : ℝ} (h : Hyperbola a b) (p : PointOnHyperbola h) (focus : ℝ) : ℝ := sorry

/-- The angle between the lines from a point to the foci -/
def angle_between_foci {a b : ℝ} (h : Hyperbola a b) (p : PointOnHyperbola h) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity {a b : ℝ} (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If there exists a point on the hyperbola where the angle between the lines to the foci is 90° and the distance to one focus is twice the distance to the other, then the eccentricity is √5 -/
theorem eccentricity_is_sqrt_five {a b : ℝ} (h : Hyperbola a b) :
  (∃ p : PointOnHyperbola h, 
    angle_between_foci h p = Real.pi / 2 ∧ 
    dist_to_focus h p (foci h).1 = 2 * dist_to_focus h p (foci h).2) →
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_eccentricity_is_sqrt_five_l1513_151302


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l1513_151374

theorem probability_four_twos_in_five_rolls (p : ℝ) (h1 : p = 1 / 6) :
  (5 : ℝ) * p^4 * (1 - p) = 5 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l1513_151374


namespace NUMINAMATH_CALUDE_detergent_calculation_l1513_151396

/-- Calculates the amount of detergent in a solution given the ratio of detergent to water and the amount of water -/
def detergent_amount (detergent_ratio : ℚ) (water_ratio : ℚ) (water_amount : ℚ) : ℚ :=
  (detergent_ratio / water_ratio) * water_amount

theorem detergent_calculation :
  let detergent_ratio : ℚ := 1
  let water_ratio : ℚ := 8
  let water_amount : ℚ := 300
  detergent_amount detergent_ratio water_ratio water_amount = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_calculation_l1513_151396


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1513_151389

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x - 1) ≥ 5 ↔ x ≤ -1 ∨ x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1513_151389


namespace NUMINAMATH_CALUDE_large_cylinder_height_l1513_151303

-- Define constants
def small_cylinder_diameter : ℝ := 3
def small_cylinder_height : ℝ := 6
def large_cylinder_diameter : ℝ := 20
def small_cylinders_to_fill : ℝ := 74.07407407407408

-- Define the theorem
theorem large_cylinder_height :
  let small_cylinder_volume := π * (small_cylinder_diameter / 2)^2 * small_cylinder_height
  let large_cylinder_radius := large_cylinder_diameter / 2
  let large_cylinder_volume := small_cylinders_to_fill * small_cylinder_volume
  large_cylinder_volume = π * large_cylinder_radius^2 * 10 := by
  sorry

end NUMINAMATH_CALUDE_large_cylinder_height_l1513_151303


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1513_151363

theorem inequality_solution_set (x : ℝ) : 
  (Set.Iio (-1) ∪ Set.Ioi 3) = {x | (3 - x) / (x + 1) < 0} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1513_151363


namespace NUMINAMATH_CALUDE_no_three_digit_base_7_equals_two_digit_base_6_l1513_151335

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if a number is representable as a two-digit number in base 6 --/
def is_two_digit_base_6 (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), d1 < 6 ∧ d2 < 6 ∧ n = to_base_10 [d1, d2] 6

theorem no_three_digit_base_7_equals_two_digit_base_6 :
  ¬ ∃ (d1 d2 d3 : ℕ), 
    d1 > 0 ∧ d1 < 7 ∧ d2 < 7 ∧ d3 < 7 ∧ 
    is_two_digit_base_6 (to_base_10 [d1, d2, d3] 7) :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_base_7_equals_two_digit_base_6_l1513_151335


namespace NUMINAMATH_CALUDE_power_product_equality_l1513_151324

theorem power_product_equality : 0.25^2015 * 4^2016 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1513_151324


namespace NUMINAMATH_CALUDE_racket_purchase_cost_l1513_151320

/-- The cost of two rackets with discounts -/
def total_cost (original_price : ℝ) : ℝ :=
  let first_racket_cost := original_price * (1 - 0.2)
  let second_racket_cost := original_price * 0.5
  first_racket_cost + second_racket_cost

/-- Theorem stating the total cost of two rackets -/
theorem racket_purchase_cost :
  total_cost 60 = 78 := by sorry

end NUMINAMATH_CALUDE_racket_purchase_cost_l1513_151320


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1513_151305

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1513_151305


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l1513_151306

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- A point bisects a chord of the ellipse -/
def bisects_chord (x y : ℝ) (px py : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_ellipse x1 y1 ∧
    is_on_ellipse x2 y2 ∧
    px = (x1 + x2) / 2 ∧
    py = (y1 + y2) / 2

/-- The equation of a line -/
def on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem ellipse_chord_theorem :
  ∀ (x y : ℝ),
    is_on_ellipse x y →
    bisects_chord x y 4 2 →
    on_line x y 1 2 (-8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l1513_151306


namespace NUMINAMATH_CALUDE_simplify_expression_l1513_151309

theorem simplify_expression (x : ℝ) : (2*x + 25) + (150*x + 35) + (50*x + 10) = 202*x + 70 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1513_151309


namespace NUMINAMATH_CALUDE_squaredigital_numbers_l1513_151380

/-- Sum of digits of a non-negative integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is squaredigital if it equals the square of the sum of its digits -/
def is_squaredigital (n : ℕ) : Prop := n = (sum_of_digits n)^2

/-- The only squaredigital numbers are 0, 1, and 81 -/
theorem squaredigital_numbers : 
  ∀ n : ℕ, is_squaredigital n ↔ n = 0 ∨ n = 1 ∨ n = 81 := by sorry

end NUMINAMATH_CALUDE_squaredigital_numbers_l1513_151380


namespace NUMINAMATH_CALUDE_four_tuple_count_l1513_151315

theorem four_tuple_count (p : ℕ) (hp : Prime p) : 
  (Finset.filter 
    (fun (t : ℕ × ℕ × ℕ × ℕ) => 
      0 < t.1 ∧ t.1 < p - 1 ∧
      0 < t.2.1 ∧ t.2.1 < p - 1 ∧
      0 < t.2.2.1 ∧ t.2.2.1 < p - 1 ∧
      0 < t.2.2.2 ∧ t.2.2.2 < p - 1 ∧
      (t.1 * t.2.2.2) % p = (t.2.1 * t.2.2.1) % p)
    (Finset.product 
      (Finset.range (p - 1)) 
      (Finset.product 
        (Finset.range (p - 1)) 
        (Finset.product 
          (Finset.range (p - 1)) 
          (Finset.range (p - 1)))))).card = (p - 1)^3 :=
by sorry


end NUMINAMATH_CALUDE_four_tuple_count_l1513_151315


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1513_151330

theorem complex_equation_solution (a : ℝ) (z : ℂ) 
  (h1 : a ≥ 0) 
  (h2 : z * Complex.abs z + a * z + Complex.I = 0) : 
  z = Complex.I * ((a - Real.sqrt (a^2 + 4)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1513_151330


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1513_151317

theorem complex_fraction_simplification :
  (7 : ℂ) + 15 * Complex.I / ((3 : ℂ) - 4 * Complex.I) = -39 / 25 + (73 / 25 : ℝ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1513_151317


namespace NUMINAMATH_CALUDE_evaluate_expression_l1513_151355

theorem evaluate_expression : (4^4 - 4*(4-2)^4)^4 = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1513_151355


namespace NUMINAMATH_CALUDE_weitzenboeck_inequality_tetrahedron_l1513_151340

/-- A tetrahedron with edge lengths a, b, c, d, e, f and surface area S. -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  S : ℝ

/-- The Weitzenböck inequality for tetrahedra. -/
theorem weitzenboeck_inequality_tetrahedron (t : Tetrahedron) :
  t.S ≤ (Real.sqrt 3 / 6) * (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) := by
  sorry

end NUMINAMATH_CALUDE_weitzenboeck_inequality_tetrahedron_l1513_151340


namespace NUMINAMATH_CALUDE_fresh_corn_processing_capacity_l1513_151367

/-- The daily processing capacity of fresh corn before technological improvement -/
def daily_capacity : ℕ := 2400

/-- The annual processing capacity before technological improvement -/
def annual_capacity : ℕ := 260000

/-- The improvement factor for daily processing capacity -/
def improvement_factor : ℚ := 13/10

/-- The reduction in processing time after improvement (in days) -/
def time_reduction : ℕ := 25

theorem fresh_corn_processing_capacity :
  daily_capacity = 2400 ∧
  annual_capacity = 260000 ∧
  (annual_capacity : ℚ) / daily_capacity - 
    (annual_capacity : ℚ) / (improvement_factor * daily_capacity) = time_reduction := by
  sorry

#check fresh_corn_processing_capacity

end NUMINAMATH_CALUDE_fresh_corn_processing_capacity_l1513_151367


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1513_151326

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The third term of an arithmetic sequence {aₙ} where a₁ + a₅ = 6 equals 3 -/
theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 5 = 6) :
  a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1513_151326


namespace NUMINAMATH_CALUDE_f_of_three_equals_zero_l1513_151331

theorem f_of_three_equals_zero (f : ℝ → ℝ) (h : ∀ x, f (1 - 2*x) = x^2 + x) : f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_zero_l1513_151331


namespace NUMINAMATH_CALUDE_sin_minus_pi_half_times_tan_pi_minus_l1513_151311

open Real

theorem sin_minus_pi_half_times_tan_pi_minus (α : ℝ) : 
  sin (α - π / 2) * tan (π - α) = sin α := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_pi_half_times_tan_pi_minus_l1513_151311


namespace NUMINAMATH_CALUDE_triangle_properties_l1513_151348

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) :
  (abc.B.cos = -5/13 ∧ 
   (2 * abc.A.sin) * (2 * abc.C.sin) = abc.B.sin^2 ∧ 
   1/2 * abc.a * abc.c * abc.B.sin = 6/13) →
  (abc.a + abc.c) / 2 = Real.sqrt 221 / 13
  ∧
  (abc.B.cos = -5/13 ∧ 
   abc.C.cos = 4/5 ∧ 
   abc.b * abc.c * abc.A.cos = 14) →
  abc.a = 11/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1513_151348


namespace NUMINAMATH_CALUDE_exists_two_students_with_not_less_scores_l1513_151398

/-- Represents a student's scores for three problems -/
structure StudentScores where
  problem1 : Fin 8
  problem2 : Fin 8
  problem3 : Fin 8

/-- Checks if one student's scores are not less than another's -/
def scoresNotLessThan (a b : StudentScores) : Prop :=
  a.problem1 ≥ b.problem1 ∧ a.problem2 ≥ b.problem2 ∧ a.problem3 ≥ b.problem3

/-- The main theorem to be proved -/
theorem exists_two_students_with_not_less_scores 
  (students : Fin 249 → StudentScores) : 
  ∃ (i j : Fin 249), i ≠ j ∧ scoresNotLessThan (students i) (students j) := by
  sorry

end NUMINAMATH_CALUDE_exists_two_students_with_not_less_scores_l1513_151398


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1513_151382

/-- The number of ways to choose a k-person committee from a group of n people -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the club -/
def club_size : ℕ := 11

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem stating that the number of ways to choose a 5-person committee from a club of 11 people is 462 -/
theorem committee_selection_ways : choose club_size committee_size = 462 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1513_151382


namespace NUMINAMATH_CALUDE_complex_magnitude_l1513_151343

theorem complex_magnitude (z : ℂ) (h : z * (1 - 2*I) = 3 + I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1513_151343


namespace NUMINAMATH_CALUDE_second_month_sale_l1513_151337

def sales_data : List ℕ := [800, 1000, 700, 800, 900]
def num_months : ℕ := 6
def average_sale : ℕ := 850

theorem second_month_sale :
  ∃ (second_month : ℕ),
    (List.sum sales_data + second_month) / num_months = average_sale ∧
    second_month = 900 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1513_151337


namespace NUMINAMATH_CALUDE_bank_deposit_problem_l1513_151392

theorem bank_deposit_problem (P : ℝ) : 
  P > 0 →
  (0.15 * P * 5 - 0.15 * P * 3.5 = 144) →
  P = 640 := by
sorry

end NUMINAMATH_CALUDE_bank_deposit_problem_l1513_151392


namespace NUMINAMATH_CALUDE_store_discount_calculation_l1513_151312

theorem store_discount_calculation (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.30 →
  additional_discount = 0.15 →
  claimed_discount = 0.45 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_both := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_both
  (actual_discount = 0.405 ∧ claimed_discount - actual_discount = 0.045) := by
  sorry

#check store_discount_calculation

end NUMINAMATH_CALUDE_store_discount_calculation_l1513_151312


namespace NUMINAMATH_CALUDE_max_value_of_operation_max_value_achieved_l1513_151391

theorem max_value_of_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 3 * (300 - 2 * n) ≤ 840 := by
sorry

theorem max_value_achieved : 
  ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - 2 * n) = 840 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_operation_max_value_achieved_l1513_151391


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l1513_151351

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 3 * m - 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧
  (A ∪ B 3 = {x : ℝ | -3 ≤ x ∧ x ≤ 8}) := by
  sorry

-- Theorem for part 2
theorem intersection_equals_B_iff_m_leq_1 (m : ℝ) :
  A ∩ B m = B m ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l1513_151351


namespace NUMINAMATH_CALUDE_total_carrots_eq_101_l1513_151388

/-- The number of carrots grown by Joan -/
def joan_carrots : ℕ := 29

/-- The number of watermelons grown by Joan -/
def joan_watermelons : ℕ := 14

/-- The number of carrots grown by Jessica -/
def jessica_carrots : ℕ := 11

/-- The number of cantaloupes grown by Jessica -/
def jessica_cantaloupes : ℕ := 9

/-- The number of carrots grown by Michael -/
def michael_carrots : ℕ := 37

/-- The number of carrots grown by Taylor -/
def taylor_carrots : ℕ := 24

/-- The number of cantaloupes grown by Taylor -/
def taylor_cantaloupes : ℕ := 3

/-- The total number of carrots grown by all -/
def total_carrots : ℕ := joan_carrots + jessica_carrots + michael_carrots + taylor_carrots

theorem total_carrots_eq_101 : total_carrots = 101 :=
by sorry

end NUMINAMATH_CALUDE_total_carrots_eq_101_l1513_151388


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1513_151316

theorem right_triangle_third_side : ∀ a b c : ℝ,
  (a^2 - 9*a + 20 = 0) →
  (b^2 - 9*b + 20 = 0) →
  (a ≠ b) →
  (a^2 + b^2 = c^2) →
  (c = 3 ∨ c = Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1513_151316


namespace NUMINAMATH_CALUDE_factory_production_difference_l1513_151393

/-- Represents the production rate and total products for a machine type -/
structure MachineType where
  rate : ℕ  -- products per minute
  total : ℕ -- total products made

/-- Calculates the difference in products between two machine types -/
def productDifference (a b : MachineType) : ℕ :=
  b.total - a.total

theorem factory_production_difference :
  let machineA : MachineType := { rate := 5, total := 25 }
  let machineB : MachineType := { rate := 8, total := 40 }
  productDifference machineA machineB = 15 := by
  sorry

#eval productDifference { rate := 5, total := 25 } { rate := 8, total := 40 }

end NUMINAMATH_CALUDE_factory_production_difference_l1513_151393


namespace NUMINAMATH_CALUDE_circle_symmetry_l1513_151336

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define circle C1
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define circle C2
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry relation between two points with respect to the line
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_of_symmetry ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ,
  (∃ x1 y1 : ℝ, circle_C1 x1 y1 ∧ symmetric_points x1 y1 x y) →
  circle_C2 x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1513_151336


namespace NUMINAMATH_CALUDE_doubled_to_original_ratio_l1513_151386

theorem doubled_to_original_ratio (x : ℝ) : 3 * (2 * x + 9) = 57 → (2 * x) / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_doubled_to_original_ratio_l1513_151386
