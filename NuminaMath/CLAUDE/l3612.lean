import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_l3612_361275

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s ^ 2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3612_361275


namespace NUMINAMATH_CALUDE_prob_rain_A_given_B_l3612_361225

/-- The probability of rain in city A given rain in city B -/
theorem prob_rain_A_given_B (pA pB pAB : ℝ) : 
  pA = 0.2 → pB = 0.18 → pAB = 0.12 → pAB / pB = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_A_given_B_l3612_361225


namespace NUMINAMATH_CALUDE_conic_section_classification_l3612_361247

/-- Given an interior angle θ of a triangle, vectors m and n, and their dot product,
    prove that the equation x²sin θ - y²cos θ = 1 represents an ellipse with foci on the y-axis. -/
theorem conic_section_classification (θ : ℝ) (m n : Fin 2 → ℝ) :
  (∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ θ = Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) →  -- θ is an interior angle of a triangle
  (m 0 = Real.sin θ ∧ m 1 = Real.cos θ) →  -- m⃗ = (sin θ, cos θ)
  (n 0 = 1 ∧ n 1 = 1) →  -- n⃗ = (1, 1)
  (m 0 * n 0 + m 1 * n 1 = 1/3) →  -- m⃗ · n⃗ = 1/3
  (∃ (x y : ℝ), x^2 * Real.sin θ - y^2 * Real.cos θ = 1) →  -- Equation: x²sin θ - y²cos θ = 1
  (∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b ∧ 
    ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b) :=  -- The equation represents an ellipse with foci on the y-axis
by sorry

end NUMINAMATH_CALUDE_conic_section_classification_l3612_361247


namespace NUMINAMATH_CALUDE_range_of_m_l3612_361240

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 2 / x + 1 / y = 1 / 3) (h_ineq : ∀ m : ℝ, x + 2 * y > m^2 - 2 * m) :
  -4 < m ∧ m < 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3612_361240


namespace NUMINAMATH_CALUDE_inequality_implies_identity_or_negation_l3612_361258

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)

/-- The main theorem stating that a function satisfying the inequality
    must be either the identity function or its negation -/
theorem inequality_implies_identity_or_negation (f : ℝ → ℝ) 
  (h : SatisfiesInequality f) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_identity_or_negation_l3612_361258


namespace NUMINAMATH_CALUDE_prime_quadratic_integer_roots_l3612_361273

theorem prime_quadratic_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x ^ 2 - p * x - 156 * p = 0 ∧ y ^ 2 - p * y - 156 * p = 0) → 
  p = 13 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_integer_roots_l3612_361273


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3612_361279

theorem consecutive_integers_sum (p q r s : ℤ) : 
  (q = p + 1 ∧ r = p + 2 ∧ s = p + 3) →  -- consecutive integers condition
  (p + s = 109) →                        -- given sum condition
  (q + r = 109) :=                       -- theorem to prove
by
  sorry


end NUMINAMATH_CALUDE_consecutive_integers_sum_l3612_361279


namespace NUMINAMATH_CALUDE_smallest_number_l3612_361263

def numbers : Finset ℚ := {5, -1/3, 0, -2}

theorem smallest_number : 
  ∀ x ∈ numbers, -2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3612_361263


namespace NUMINAMATH_CALUDE_bus_passengers_l3612_361293

theorem bus_passengers (men women : ℕ) : 
  women = men / 2 →
  men - 16 = women + 8 →
  men + women = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l3612_361293


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3612_361291

/-- Triangle ABC with vertices A(-3,0), B(2,1), and C(-2,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def ABC : Triangle := ⟨(-3, 0), (2, 1), (-2, 3)⟩

/-- The equation of line BC -/
def line_BC : LineEquation := ⟨1, 2, -4⟩

/-- The equation of the perpendicular bisector of BC -/
def perp_bisector_BC : LineEquation := ⟨2, -1, 2⟩

theorem triangle_ABC_properties :
  let t := ABC
  (line_BC.a * t.B.1 + line_BC.b * t.B.2 + line_BC.c = 0 ∧
   line_BC.a * t.C.1 + line_BC.b * t.C.2 + line_BC.c = 0) ∧
  (perp_bisector_BC.a * ((t.B.1 + t.C.1) / 2) + 
   perp_bisector_BC.b * ((t.B.2 + t.C.2) / 2) + 
   perp_bisector_BC.c = 0 ∧
   perp_bisector_BC.a * line_BC.b = -perp_bisector_BC.b * line_BC.a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3612_361291


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l3612_361239

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The property that f(x^2) = f(f(x)) = (f(x))^2 for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x^2) = f (f x) ∧ f (x^2) = (f x)^2

theorem unique_quadratic_function :
  ∃! f : ℝ → ℝ, QuadraticFunction f ∧ SatisfiesCondition f ∧ ∀ x, f x = x^2 :=
by
  sorry

#check unique_quadratic_function

end NUMINAMATH_CALUDE_unique_quadratic_function_l3612_361239


namespace NUMINAMATH_CALUDE_quadratic_properties_l3612_361257

/-- The quadratic function f(x) = 2x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 3

theorem quadratic_properties :
  (∀ x y : ℝ, y = f x → y > f (-1) → x ≠ -1) ∧ 
  (f (-1) = -5) ∧
  (∀ x : ℝ, -2 ≤ x → x ≤ 1 → -5 ≤ f x ∧ f x ≤ 3) ∧
  (∀ x y : ℝ, y = 2 * (x - 1)^2 - 4 ↔ y = f (x - 2) + 1) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_properties_l3612_361257


namespace NUMINAMATH_CALUDE_medicine_tablets_l3612_361282

theorem medicine_tablets (tablets_A tablets_B min_extraction : ℕ) : 
  tablets_A = 10 →
  min_extraction = 15 →
  min_extraction = tablets_A + 2 + (tablets_B - 2) →
  tablets_B = 5 := by
sorry

end NUMINAMATH_CALUDE_medicine_tablets_l3612_361282


namespace NUMINAMATH_CALUDE_company_workers_count_l3612_361287

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  supervisors : ℕ
  teamLeadsPerSupervisor : ℕ
  workersPerTeamLead : ℕ

/-- Calculates the total number of workers in a company given its hierarchy -/
def totalWorkers (c : CompanyHierarchy) : ℕ :=
  c.supervisors * c.teamLeadsPerSupervisor * c.workersPerTeamLead

/-- Theorem stating that a company with 13 supervisors, 3 team leads per supervisor,
    and 10 workers per team lead has 390 workers in total -/
theorem company_workers_count :
  let c : CompanyHierarchy := {
    supervisors := 13,
    teamLeadsPerSupervisor := 3,
    workersPerTeamLead := 10
  }
  totalWorkers c = 390 := by
  sorry


end NUMINAMATH_CALUDE_company_workers_count_l3612_361287


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l3612_361264

/-- The probability of getting at least one question right when randomly guessing 5 questions,
    each with 6 answer choices. -/
theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 5 → choices = 6 → (1 - (choices - 1 : ℚ) / choices ^ n) = 4651 / 7776 := by
  sorry

#check probability_at_least_one_correct

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l3612_361264


namespace NUMINAMATH_CALUDE_music_school_enrollment_cost_l3612_361266

/-- Calculates the total cost for music school enrollment for four siblings --/
theorem music_school_enrollment_cost :
  let regular_tuition : ℕ := 45
  let early_bird_discount : ℕ := 15
  let first_sibling_discount : ℕ := 15
  let additional_sibling_discount : ℕ := 10
  let weekend_class_extra : ℕ := 20
  let multi_instrument_discount : ℕ := 10

  let ali_cost : ℕ := regular_tuition - early_bird_discount
  let matt_cost : ℕ := regular_tuition - first_sibling_discount
  let jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra - multi_instrument_discount
  let sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra - multi_instrument_discount

  ali_cost + matt_cost + jane_cost + sarah_cost = 150 :=
by
  sorry


end NUMINAMATH_CALUDE_music_school_enrollment_cost_l3612_361266


namespace NUMINAMATH_CALUDE_fraction_equality_l3612_361261

theorem fraction_equality (a b c x : ℝ) 
  (hx : x = a / b) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) : 
  (a + 2*b + 3*c) / (a - b - 3*c) = (b*(x + 2) + 3*c) / (b*(x - 1) - 3*c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3612_361261


namespace NUMINAMATH_CALUDE_jim_distance_l3612_361231

/-- Represents the distance covered by a person in a certain number of steps -/
structure StepDistance where
  steps : ℕ
  distance : ℝ

/-- Carly's step distance -/
def carly_step : ℝ := 0.5

/-- The relationship between Carly's and Jim's steps for the same distance -/
def step_ratio : ℚ := 3 / 4

/-- Number of Jim's steps we want to calculate the distance for -/
def jim_steps : ℕ := 24

/-- Theorem stating that Jim travels 9 metres in 24 steps -/
theorem jim_distance : 
  ∀ (carly : StepDistance) (jim : StepDistance),
  carly.steps = 3 ∧ 
  jim.steps = 4 ∧
  carly.distance = jim.distance ∧
  carly.distance = carly_step * carly.steps →
  (jim_steps : ℝ) * jim.distance / jim.steps = 9 := by
  sorry

end NUMINAMATH_CALUDE_jim_distance_l3612_361231


namespace NUMINAMATH_CALUDE_yogurt_satisfaction_probability_l3612_361205

theorem yogurt_satisfaction_probability 
  (total_sample : ℕ) 
  (satisfied_with_yogurt : ℕ) 
  (h1 : total_sample = 500) 
  (h2 : satisfied_with_yogurt = 370) : 
  (satisfied_with_yogurt : ℚ) / total_sample = 37 / 50 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_satisfaction_probability_l3612_361205


namespace NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l3612_361212

/-- The meeting point of a cyclist and a pedestrian on a straight path --/
theorem cyclist_pedestrian_meeting_point (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let total_distance := a + b
  let cyclist_speed := total_distance
  let pedestrian_speed := a
  let meeting_point := a * (a + b) / (2 * a + b)
  meeting_point / cyclist_speed = (a - meeting_point) / pedestrian_speed ∧
  meeting_point < a :=
by sorry

end NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l3612_361212


namespace NUMINAMATH_CALUDE_dad_steps_l3612_361219

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between steps taken by Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

/-- Theorem stating that given the conditions, Dad took 90 steps -/
theorem dad_steps :
  ∀ s : Steps,
  dad_masha_ratio s →
  masha_yasha_ratio s →
  total_masha_yasha s →
  s.dad = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l3612_361219


namespace NUMINAMATH_CALUDE_combined_original_price_l3612_361224

/-- Proves that the combined original price of a candy box, a can of soda, and a bag of chips
    was 34 pounds, given their new prices after specific percentage increases. -/
theorem combined_original_price (candy_new : ℝ) (soda_new : ℝ) (chips_new : ℝ)
    (h_candy : candy_new = 20)
    (h_soda : soda_new = 6)
    (h_chips : chips_new = 8)
    (h_candy_increase : candy_new = (5/4) * (candy_new - (1/4) * candy_new))
    (h_soda_increase : soda_new = (3/2) * (soda_new - (1/2) * soda_new))
    (h_chips_increase : chips_new = (11/10) * (chips_new - (1/10) * chips_new)) :
  (candy_new - (1/4) * candy_new) + (soda_new - (1/2) * soda_new) + (chips_new - (1/10) * chips_new) = 34 := by
  sorry

end NUMINAMATH_CALUDE_combined_original_price_l3612_361224


namespace NUMINAMATH_CALUDE_common_root_condition_rational_roots_if_common_root_l3612_361271

structure QuadraticEquation (α : Type) [Field α] where
  p : α
  q : α

def hasCommonRoot {α : Type} [Field α] (eq1 eq2 : QuadraticEquation α) : Prop :=
  ∃ x : α, x^2 + eq1.p * x + eq1.q = 0 ∧ x^2 + eq2.p * x + eq2.q = 0

theorem common_root_condition {α : Type} [Field α] (eq1 eq2 : QuadraticEquation α) :
  hasCommonRoot eq1 eq2 ↔ (eq1.p - eq2.p) * (eq1.p * eq2.q - eq2.p * eq1.q) + (eq1.q - eq2.q)^2 = 0 :=
sorry

theorem rational_roots_if_common_root (eq1 eq2 : QuadraticEquation ℚ) 
  (h1 : hasCommonRoot eq1 eq2) (h2 : eq1 ≠ eq2) :
  ∃ (x y : ℚ), (x^2 + eq1.p * x + eq1.q = 0 ∧ y^2 + eq1.p * y + eq1.q = 0) ∧
                (x^2 + eq2.p * x + eq2.q = 0 ∧ y^2 + eq2.p * y + eq2.q = 0) :=
sorry

end NUMINAMATH_CALUDE_common_root_condition_rational_roots_if_common_root_l3612_361271


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l3612_361207

/-- The number of unit squares in the nth ring of a square array with a center square,
    where each ring increases by 3 on each side. -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 50th ring contains 408 unit squares -/
theorem fiftieth_ring_squares : ring_squares 50 = 408 := by
  sorry

#eval ring_squares 50  -- This will evaluate to 408

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l3612_361207


namespace NUMINAMATH_CALUDE_f_r_correct_l3612_361268

/-- The number of ways to select k elements from a permutation of n elements,
    such that any two selected elements are separated by at least r elements
    in the original permutation. -/
def f_r (n k r : ℕ) : ℕ :=
  Nat.choose (n - k * r + r) k

/-- Theorem stating that f_r(n, k, r) correctly counts the number of ways to select
    k elements from a permutation of n elements with the given separation condition. -/
theorem f_r_correct (n k r : ℕ) :
  f_r n k r = Nat.choose (n - k * r + r) k :=
by sorry

end NUMINAMATH_CALUDE_f_r_correct_l3612_361268


namespace NUMINAMATH_CALUDE_hospital_current_age_l3612_361297

/-- Represents the current age of Grant -/
def grants_current_age : ℕ := 25

/-- Represents the number of years in the future when the condition is met -/
def years_in_future : ℕ := 5

/-- Represents the fraction of the hospital's age that Grant will be in the future -/
def age_fraction : ℚ := 2/3

/-- Theorem stating that given the conditions, the current age of the hospital is 40 years -/
theorem hospital_current_age : 
  ∃ (hospital_age : ℕ), 
    (grants_current_age + years_in_future : ℚ) = age_fraction * (hospital_age + years_in_future : ℚ) ∧
    hospital_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_hospital_current_age_l3612_361297


namespace NUMINAMATH_CALUDE_mart_income_percentage_of_juan_l3612_361200

/-- Represents the income relationships between Tim, Mart, Juan, and Alex -/
structure IncomeRelationships where
  tim : ℝ
  mart : ℝ
  juan : ℝ
  alex : ℝ
  mart_tim_ratio : mart = 1.6 * tim
  tim_juan_ratio : tim = 0.6 * juan
  alex_mart_ratio : alex = 1.25 * mart
  juan_alex_ratio : juan = 1.2 * alex

/-- Theorem stating that Mart's income is 96% of Juan's income -/
theorem mart_income_percentage_of_juan (ir : IncomeRelationships) :
  ir.mart = 0.96 * ir.juan := by
  sorry

end NUMINAMATH_CALUDE_mart_income_percentage_of_juan_l3612_361200


namespace NUMINAMATH_CALUDE_matchstick_triangle_solutions_l3612_361299

/-- Represents a triangle formed by matchsticks -/
structure MatchstickTriangle where
  shortest : ℕ
  middle : ℕ
  longest : ℕ

/-- Checks if a MatchstickTriangle is valid according to the problem conditions -/
def isValidTriangle (t : MatchstickTriangle) : Prop :=
  t.shortest + t.middle + t.longest = 100 ∧
  t.longest = 3 * t.shortest ∧
  t.shortest < t.middle ∧
  t.middle < t.longest

theorem matchstick_triangle_solutions :
  ∀ t : MatchstickTriangle, isValidTriangle t →
    (t.shortest = 15 ∧ t.middle = 40 ∧ t.longest = 45) ∨
    (t.shortest = 16 ∧ t.middle = 36 ∧ t.longest = 48) :=
by sorry

end NUMINAMATH_CALUDE_matchstick_triangle_solutions_l3612_361299


namespace NUMINAMATH_CALUDE_triangle_k_range_l3612_361243

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three lines form a triangle -/
def form_triangle (l₁ l₂ l₃ : Line) : Prop :=
  ∀ (x y : ℝ), (l₁.a * x + l₁.b * y + l₁.c = 0 ∧ 
                l₂.a * x + l₂.b * y + l₂.c = 0 ∧
                l₃.a * x + l₃.b * y + l₃.c = 0) → False

/-- The theorem stating the range of k for which the given lines form a triangle -/
theorem triangle_k_range :
  ∀ (k : ℝ),
  let l₁ : Line := ⟨1, -1, 0⟩
  let l₂ : Line := ⟨1, 1, -2⟩
  let l₃ : Line := ⟨5, -k, -15⟩
  form_triangle l₁ l₂ l₃ ↔ k ≠ 5 ∧ k ≠ -5 ∧ k ≠ -10 :=
sorry

end NUMINAMATH_CALUDE_triangle_k_range_l3612_361243


namespace NUMINAMATH_CALUDE_chinese_coin_problem_l3612_361227

/-- Represents an arithmetic sequence of 5 terms -/
structure ArithmeticSequence :=
  (a : ℚ) -- First term
  (d : ℚ) -- Common difference

/-- Properties of the specific arithmetic sequence in the problem -/
def ProblemSequence (seq : ArithmeticSequence) : Prop :=
  -- Sum of all terms is 5
  seq.a - 2*seq.d + seq.a - seq.d + seq.a + seq.a + seq.d + seq.a + 2*seq.d = 5 ∧
  -- Sum of first two terms equals sum of last three terms
  seq.a - 2*seq.d + seq.a - seq.d = seq.a + seq.a + seq.d + seq.a + 2*seq.d

theorem chinese_coin_problem (seq : ArithmeticSequence) 
  (h : ProblemSequence seq) : seq.a - seq.d = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_chinese_coin_problem_l3612_361227


namespace NUMINAMATH_CALUDE_min_sum_p_q_l3612_361285

theorem min_sum_p_q (p q : ℕ) (hp : p > 1) (hq : q > 1) (h_eq : 17 * (p + 1) = 20 * (q + 1)) :
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 20 * (q' + 1) → p + q ≤ p' + q' ∧ p + q = 37 := by
sorry

end NUMINAMATH_CALUDE_min_sum_p_q_l3612_361285


namespace NUMINAMATH_CALUDE_intersection_locus_l3612_361262

/-- Given two fixed points A(a, 0) and B(b, 0) on the x-axis, and a moving point C(0, c) on the y-axis,
    prove that the locus of the intersection point of line BC and line l (which passes through the origin
    and is perpendicular to AC) satisfies the equation (x - b/2)²/(b²/4) + y²/(ab/4) = 1 -/
theorem intersection_locus (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ∃ (x y : ℝ → ℝ), ∀ (c : ℝ),
    let l := {p : ℝ × ℝ | p.2 = (a / c) * p.1}
    let bc := {p : ℝ × ℝ | p.1 / b + p.2 / c = 1}
    let intersection := Set.inter l bc
    (x c, y c) ∈ intersection ∧
    (x c - b/2)^2 / (b^2/4) + (y c)^2 / (a*b/4) = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_locus_l3612_361262


namespace NUMINAMATH_CALUDE_four_color_plane_partition_l3612_361237

-- Define the plane as ℝ × ℝ
def Plane := ℝ × ℝ

-- Define a partition of the plane into four subsets
def Partition (A B C D : Set Plane) : Prop :=
  (A ∪ B ∪ C ∪ D = Set.univ) ∧
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (A ∩ D = ∅) ∧
  (B ∩ C = ∅) ∧ (B ∩ D = ∅) ∧ (C ∩ D = ∅)

-- Define a circle in the plane
def Circle (center : Plane) (radius : ℝ) : Set Plane :=
  {p : Plane | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem four_color_plane_partition :
  ∃ (A B C D : Set Plane), Partition A B C D ∧
    ∀ (center : Plane) (radius : ℝ),
      (Circle center radius ∩ A).Nonempty ∧
      (Circle center radius ∩ B).Nonempty ∧
      (Circle center radius ∩ C).Nonempty ∧
      (Circle center radius ∩ D).Nonempty :=
by sorry


end NUMINAMATH_CALUDE_four_color_plane_partition_l3612_361237


namespace NUMINAMATH_CALUDE_log_27_3_l3612_361223

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l3612_361223


namespace NUMINAMATH_CALUDE_angle_sum_from_tangent_roots_l3612_361251

theorem angle_sum_from_tangent_roots (α β : Real) :
  (∃ x y : Real, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ 
                 y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
                 x = Real.tan α ∧ 
                 y = Real.tan β) →
  -π/2 < α ∧ α < π/2 →
  -π/2 < β ∧ β < π/2 →
  α + β = -2*π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_from_tangent_roots_l3612_361251


namespace NUMINAMATH_CALUDE_product_sum_squares_l3612_361201

theorem product_sum_squares (x y : ℝ) :
  x * y = 120 ∧ x^2 + y^2 = 289 → x + y = 22 ∨ x + y = -22 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_squares_l3612_361201


namespace NUMINAMATH_CALUDE_rain_probability_both_days_l3612_361272

theorem rain_probability_both_days 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (prob_sunday_given_saturday : ℝ) 
  (h1 : prob_saturday = 0.4)
  (h2 : prob_sunday = 0.3)
  (h3 : prob_sunday_given_saturday = 0.5) :
  prob_saturday * prob_sunday_given_saturday = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_both_days_l3612_361272


namespace NUMINAMATH_CALUDE_seconds_in_day_l3612_361298

/-- Represents the number of minutes in a day on the island of Misfortune -/
def minutes_per_day : ℕ := 77

/-- Represents the number of seconds in an hour on the island of Misfortune -/
def seconds_per_hour : ℕ := 91

/-- Theorem stating that there are 1001 seconds in a day on the island of Misfortune -/
theorem seconds_in_day : 
  ∃ (hours_per_day minutes_per_hour seconds_per_minute : ℕ), 
    hours_per_day * minutes_per_hour = minutes_per_day ∧
    minutes_per_hour * seconds_per_minute = seconds_per_hour ∧
    hours_per_day * minutes_per_hour * seconds_per_minute = 1001 :=
by
  sorry


end NUMINAMATH_CALUDE_seconds_in_day_l3612_361298


namespace NUMINAMATH_CALUDE_find_number_l3612_361216

theorem find_number : ∃ x : ℕ,
  x % 18 = 6 ∧
  190 % 18 = 10 ∧
  x < 190 ∧
  (∀ y : ℕ, y % 18 = 6 → y < 190 → y ≤ x) ∧
  x = 186 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3612_361216


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3612_361288

-- Define the sets P and Q
def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

-- State the theorem
theorem set_intersection_theorem : (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3612_361288


namespace NUMINAMATH_CALUDE_bike_wheel_radius_increase_l3612_361246

theorem bike_wheel_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 30)
  (h2 : final_circumference = 40) :
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_bike_wheel_radius_increase_l3612_361246


namespace NUMINAMATH_CALUDE_complex_expansion_l3612_361218

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_expansion : i * (1 + i)^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expansion_l3612_361218


namespace NUMINAMATH_CALUDE_magnitude_ratio_not_sufficient_for_parallel_l3612_361286

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem magnitude_ratio_not_sufficient_for_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ (a b : V), ‖a‖ = 2 * ‖b‖ → parallel a b) := by
  sorry


end NUMINAMATH_CALUDE_magnitude_ratio_not_sufficient_for_parallel_l3612_361286


namespace NUMINAMATH_CALUDE_two_fifths_in_twice_one_tenth_l3612_361226

theorem two_fifths_in_twice_one_tenth : (2 * (1 / 10)) / (2 / 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_twice_one_tenth_l3612_361226


namespace NUMINAMATH_CALUDE_gcd_1443_999_l3612_361230

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1443_999_l3612_361230


namespace NUMINAMATH_CALUDE_simplify_expression_l3612_361267

theorem simplify_expression (a b : ℝ) :
  (15 * a + 45 * b) + (12 * a + 35 * b) - (7 * a + 30 * b) - (3 * a + 15 * b) = 17 * a + 35 * b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3612_361267


namespace NUMINAMATH_CALUDE_henry_tournament_points_l3612_361265

/-- Point system for the tic-tac-toe tournament --/
structure PointSystem where
  win_points : ℕ
  loss_points : ℕ
  draw_points : ℕ

/-- Results of Henry's tournament --/
structure TournamentResults where
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- Calculate the total points for a given point system and tournament results --/
def calculateTotalPoints (ps : PointSystem) (tr : TournamentResults) : ℕ :=
  ps.win_points * tr.wins + ps.loss_points * tr.losses + ps.draw_points * tr.draws

/-- Theorem: Henry's total points in the tournament --/
theorem henry_tournament_points :
  let ps : PointSystem := { win_points := 5, loss_points := 2, draw_points := 3 }
  let tr : TournamentResults := { wins := 2, losses := 2, draws := 10 }
  calculateTotalPoints ps tr = 44 := by
  sorry


end NUMINAMATH_CALUDE_henry_tournament_points_l3612_361265


namespace NUMINAMATH_CALUDE_b_savings_l3612_361228

/-- Proves that given the conditions, b saves 1600 per month -/
theorem b_savings (
  income_ratio : ℚ → ℚ → Prop)  -- income ratio of a to b
  (expense_ratio : ℚ → ℚ → Prop)  -- expense ratio of a to b
  (a_savings : ℚ)  -- a's savings
  (b_income : ℚ)  -- b's income
  (h1 : income_ratio (5/11) (6/11))  -- income ratio condition
  (h2 : expense_ratio (3/7) (4/7))  -- expense ratio condition
  (h3 : a_savings = 1800)  -- a's savings condition
  (h4 : b_income = 7200)  -- b's income condition
  : ℚ  -- b's savings
  :=
by
  sorry

#check b_savings

end NUMINAMATH_CALUDE_b_savings_l3612_361228


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3612_361245

/-- Given a triangle PQR and parallel lines m_P, m_Q, m_R, find the perimeter of the triangle formed by these lines -/
theorem triangle_perimeter (PQ QR PR : ℝ) (m_P m_Q m_R : ℝ) : 
  PQ = 150 → QR = 270 → PR = 210 →
  m_P = 75 → m_Q = 60 → m_R = 30 →
  ∃ (perimeter : ℝ), abs (perimeter - 239.314) < 0.001 :=
by sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l3612_361245


namespace NUMINAMATH_CALUDE_milk_container_problem_l3612_361208

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1264

/-- The fraction of milk in container B relative to A's capacity --/
def fraction_in_B : ℝ := 0.375

/-- The amount transferred from C to B --/
def transfer_amount : ℝ := 158

theorem milk_container_problem :
  (fraction_in_B * initial_quantity + transfer_amount = 
   (1 - fraction_in_B) * initial_quantity - transfer_amount) ∧
  (initial_quantity > 0) := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l3612_361208


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3612_361277

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2*x - 7 = 0 ↔ (x - 1)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3612_361277


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3612_361209

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3612_361209


namespace NUMINAMATH_CALUDE_base_9_to_3_conversion_l3612_361295

def to_base_3 (n : ℕ) : ℕ := sorry

def from_base_9 (n : ℕ) : ℕ := sorry

theorem base_9_to_3_conversion :
  to_base_3 (from_base_9 745) = 211112 := by sorry

end NUMINAMATH_CALUDE_base_9_to_3_conversion_l3612_361295


namespace NUMINAMATH_CALUDE_tan_product_simplification_l3612_361256

theorem tan_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_simplification_l3612_361256


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l3612_361278

def y : ℕ := 2^10 * 3^15 * 4^20 * 5^25 * 6^30 * 7^35 * 8^40 * 9^45

theorem smallest_square_multiplier (n : ℕ) : 
  (∃ m : ℕ, n * y = m^2) ∧ 
  (∀ k : ℕ, k < n → ¬∃ m : ℕ, k * y = m^2) ↔ 
  n = 105 :=
sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l3612_361278


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3612_361294

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 7 ∧ x₁^2 - 4*x₁ + 7 = 10) ∧
  (x₂ = 2 - Real.sqrt 7 ∧ x₂^2 - 4*x₂ + 7 = 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3612_361294


namespace NUMINAMATH_CALUDE_replacement_cost_theorem_l3612_361221

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  short_side : ℝ
  long_side : ℝ
  perimeter : ℝ
  cost_per_foot : ℝ
  long_side_relation : long_side = 3 * short_side
  perimeter_equation : perimeter = 2 * short_side + 2 * long_side

/-- The cost to replace one short side of the fence -/
def replacement_cost (plot : RectangularPlot) : ℝ :=
  plot.short_side * plot.cost_per_foot

/-- Theorem stating the replacement cost for the given conditions -/
theorem replacement_cost_theorem (plot : RectangularPlot) 
  (h_perimeter : plot.perimeter = 640)
  (h_cost : plot.cost_per_foot = 5) :
  replacement_cost plot = 400 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_theorem_l3612_361221


namespace NUMINAMATH_CALUDE_angle_bisector_slope_l3612_361206

theorem angle_bisector_slope :
  let line1 : ℝ → ℝ := λ x => 2 * x
  let line2 : ℝ → ℝ := λ x => -2 * x
  let slope1 : ℝ := 2
  let slope2 : ℝ := -2
  let angle_bisector_slope : ℝ := (slope1 + slope2 + Real.sqrt (1 + slope1^2 + slope2^2)) / (1 - slope1 * slope2)
  angle_bisector_slope = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_slope_l3612_361206


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_eight_or_nine_l3612_361210

theorem three_digit_numbers_with_eight_or_nine (total_three_digit : ℕ) (without_eight_or_nine : ℕ) :
  total_three_digit = 900 →
  without_eight_or_nine = 448 →
  total_three_digit - without_eight_or_nine = 452 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_eight_or_nine_l3612_361210


namespace NUMINAMATH_CALUDE_work_completion_time_l3612_361254

/-- 
Given:
- A person B can do a work in 20 days
- Persons A and B together can do the work in 15 days

Prove that A can do the work alone in 60 days
-/
theorem work_completion_time (b_time : ℝ) (ab_time : ℝ) (a_time : ℝ) : 
  b_time = 20 → ab_time = 15 → a_time = 60 → 
  1 / a_time + 1 / b_time = 1 / ab_time := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3612_361254


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l3612_361233

theorem sunzi_wood_measurement_problem (x y : ℝ) :
  (x - y = 4.5 ∧ (1/2) * x + 1 = y) ↔
  (x - y = 4.5 ∧ ∃ (z : ℝ), z = x/2 ∧ z + 1 = y ∧ x - (z + 1) = 4.5) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l3612_361233


namespace NUMINAMATH_CALUDE_largest_angle_right_isosceles_triangle_l3612_361241

theorem largest_angle_right_isosceles_triangle (D E F : Real) :
  -- Triangle DEF is a right isosceles triangle
  D + E + F = 180 →
  D = E →
  (D = 90 ∨ E = 90 ∨ F = 90) →
  -- Angle D measures 45°
  D = 45 →
  -- The largest interior angle measures 90°
  max D (max E F) = 90 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_right_isosceles_triangle_l3612_361241


namespace NUMINAMATH_CALUDE_prob_at_least_one_prob_exactly_one_l3612_361220

/-- Probability of event A occurring -/
def probA : ℚ := 4/5

/-- Probability of event B occurring -/
def probB : ℚ := 3/5

/-- Probability of event C occurring -/
def probC : ℚ := 2/5

/-- Events A, B, and C are independent -/
axiom independence : True

/-- Probability of at least one event occurring -/
theorem prob_at_least_one : 
  1 - (1 - probA) * (1 - probB) * (1 - probC) = 119/125 := by sorry

/-- Probability of exactly one event occurring -/
theorem prob_exactly_one :
  probA * (1 - probB) * (1 - probC) + 
  (1 - probA) * probB * (1 - probC) + 
  (1 - probA) * (1 - probB) * probC = 37/125 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_prob_exactly_one_l3612_361220


namespace NUMINAMATH_CALUDE_sachin_age_l3612_361289

theorem sachin_age (sachin rahul : ℝ) 
  (h1 : sachin = rahul - 9)
  (h2 : sachin / rahul = 7 / 9) : 
  sachin = 31.5 := by
sorry

end NUMINAMATH_CALUDE_sachin_age_l3612_361289


namespace NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l3612_361292

-- Define the set A
def A : Set ℕ := {x | ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a ≠ 0 ∧ b ≠ 0}

-- State the theorem
theorem prime_square_in_A_implies_prime_in_A (p : ℕ) (h_prime : Nat.Prime p) :
  p^2 ∈ A → p ∈ A := by
  sorry

end NUMINAMATH_CALUDE_prime_square_in_A_implies_prime_in_A_l3612_361292


namespace NUMINAMATH_CALUDE_bryans_book_collection_l3612_361215

theorem bryans_book_collection (total_books : ℕ) (num_continents : ℕ) 
  (h1 : total_books = 488) 
  (h2 : num_continents = 4) 
  (h3 : total_books % num_continents = 0) : 
  total_books / num_continents = 122 := by
sorry

end NUMINAMATH_CALUDE_bryans_book_collection_l3612_361215


namespace NUMINAMATH_CALUDE_carnation_percentage_l3612_361284

/-- Represents a flower bouquet with various types of flowers -/
structure Bouquet where
  total : ℕ
  pink_roses : ℕ
  red_roses : ℕ
  pink_carnations : ℕ
  red_carnations : ℕ
  yellow_tulips : ℕ

/-- Conditions for the flower bouquet -/
def validBouquet (b : Bouquet) : Prop :=
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations + b.yellow_tulips = b.total ∧
  b.pink_roses + b.pink_carnations = b.total / 2 ∧
  b.pink_roses = (b.pink_roses + b.pink_carnations) * 2 / 5 ∧
  b.red_carnations = (b.red_roses + b.red_carnations) * 6 / 7 ∧
  b.yellow_tulips = b.total / 5

/-- Theorem stating that for a valid bouquet, 55% of the flowers are carnations -/
theorem carnation_percentage (b : Bouquet) (h : validBouquet b) :
  (b.pink_carnations + b.red_carnations) * 100 / b.total = 55 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_l3612_361284


namespace NUMINAMATH_CALUDE_prob_greater_than_four_l3612_361214

-- Define a fair 6-sided die
def fair_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the probability of an event on a fair die
def prob (event : Finset ℕ) : ℚ :=
  (event ∩ fair_die).card / fair_die.card

-- Define the event of rolling a number greater than 4
def greater_than_four : Finset ℕ := {5, 6}

-- Theorem statement
theorem prob_greater_than_four :
  prob greater_than_four = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_greater_than_four_l3612_361214


namespace NUMINAMATH_CALUDE_quadratic_root_sum_squares_l3612_361213

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 2*h*r + 2 = 0 ∧ s^2 + 2*h*s + 2 = 0 ∧ r^2 + s^2 = 8) → 
  |h| = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_squares_l3612_361213


namespace NUMINAMATH_CALUDE_range_of_f_range_of_g_l3612_361217

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Theorem 1: The range of f is [0,+∞) iff a = -1 or a = 3/2
theorem range_of_f (a : ℝ) : 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ 
  a = -1 ∨ a = 3/2 :=
sorry

-- Theorem 2: When f(x) ≥ 0 for all x, the range of g(a) is [-19/4, 4]
theorem range_of_g : 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → 
    (∀ y : ℝ, -19/4 ≤ y ∧ y ≤ 4 → ∃ a : ℝ, g a = y) ∧ 
    (∀ a : ℝ, -19/4 ≤ g a ∧ g a ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_g_l3612_361217


namespace NUMINAMATH_CALUDE_box_volume_problem_l3612_361203

theorem box_volume_problem :
  ∃! (x : ℕ), x > 3 ∧ (x + 3) * (x - 3) * (x^2 + 9) < 500 := by sorry

end NUMINAMATH_CALUDE_box_volume_problem_l3612_361203


namespace NUMINAMATH_CALUDE_distribute_balls_result_l3612_361290

/-- The number of ways to distribute balls to students -/
def distribute_balls (red black white : ℕ) : ℕ :=
  let min_boys := 2
  let min_girl := 3
  let remaining_red := red - (2 * min_boys + min_girl)
  let remaining_black := black - (2 * min_boys + min_girl)
  let remaining_white := white - (2 * min_boys + min_girl)
  (Nat.choose (remaining_red + 2) 2) *
  (Nat.choose (remaining_black + 2) 2) *
  (Nat.choose (remaining_white + 2) 2)

/-- Theorem stating the number of ways to distribute the balls -/
theorem distribute_balls_result : distribute_balls 10 15 20 = 47250 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_result_l3612_361290


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l3612_361248

/-- The number of ways to arrange 8 students and 2 teachers in a line,
    where the teachers cannot stand next to each other -/
def arrangement_count : ℕ :=
  Nat.factorial 8 * 9 * 8

/-- Theorem stating that the number of valid arrangements is correct -/
theorem correct_arrangement_count :
  arrangement_count = Nat.factorial 8 * 9 * 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l3612_361248


namespace NUMINAMATH_CALUDE_no_real_solution_ffx_l3612_361236

/-- A second-degree polynomial function -/
def SecondDegreePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- No real solution for f(x) = x -/
def NoRealSolutionForFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

/-- No real solution for f(f(x)) = x -/
def NoRealSolutionForFFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) ≠ x

theorem no_real_solution_ffx 
  (f : ℝ → ℝ) 
  (h1 : SecondDegreePolynomial f) 
  (h2 : NoRealSolutionForFX f) : 
  NoRealSolutionForFFX f :=
sorry

end NUMINAMATH_CALUDE_no_real_solution_ffx_l3612_361236


namespace NUMINAMATH_CALUDE_range_of_m_l3612_361232

theorem range_of_m (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 1) 
  (hab : a + b = 2) 
  (h_ineq : ∀ m, (4/a) + (1/(b-1)) > m^2 + 8*m) :
  -9 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3612_361232


namespace NUMINAMATH_CALUDE_justin_tim_same_game_l3612_361252

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of players to be selected (excluding Justin and Tim) -/
def players_to_select : ℕ := players_per_game - 2

/-- The number of remaining players after excluding Justin and Tim -/
def remaining_players : ℕ := total_players - 2

/-- The number of times Justin and Tim play in the same game -/
def same_game_count : ℕ := Nat.choose remaining_players players_to_select

theorem justin_tim_same_game :
  same_game_count = 210 := by sorry

end NUMINAMATH_CALUDE_justin_tim_same_game_l3612_361252


namespace NUMINAMATH_CALUDE_divisor_problem_l3612_361269

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_problem :
  (∃! k : ℕ, 2 ∣ k ∧ 9 ∣ k ∧ divisor_count k = 14) ∧
  (∃ k₁ k₂ : ℕ, k₁ ≠ k₂ ∧ 2 ∣ k₁ ∧ 9 ∣ k₁ ∧ divisor_count k₁ = 15 ∧
                2 ∣ k₂ ∧ 9 ∣ k₂ ∧ divisor_count k₂ = 15) ∧
  (¬ ∃ k : ℕ, 2 ∣ k ∧ 9 ∣ k ∧ divisor_count k = 17) :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3612_361269


namespace NUMINAMATH_CALUDE_work_completion_time_l3612_361204

-- Define the rates of work for A and B
def rate_A : ℚ := 1 / 16
def rate_B : ℚ := rate_A / 3

-- Define the total rate when A and B work together
def total_rate : ℚ := rate_A + rate_B

-- Theorem statement
theorem work_completion_time :
  (1 : ℚ) / total_rate = 12 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3612_361204


namespace NUMINAMATH_CALUDE_function_not_monotonic_iff_a_in_range_l3612_361253

/-- The function f(x) is not monotonic on the interval (0, 4) if and only if a is in the open interval (-4, 9/4) -/
theorem function_not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, x ∈ (Set.Ioo 0 4) ∧ y ∈ (Set.Ioo 0 4) ∧ x < y ∧
    ((1/3 * x^3 - 3/2 * x^2 + a*x + 4) > (1/3 * y^3 - 3/2 * y^2 + a*y + 4) ∨
     (1/3 * x^3 - 3/2 * x^2 + a*x + 4) < (1/3 * y^3 - 3/2 * y^2 + a*y + 4)))
  ↔ a ∈ Set.Ioo (-4) (9/4) :=
by sorry

end NUMINAMATH_CALUDE_function_not_monotonic_iff_a_in_range_l3612_361253


namespace NUMINAMATH_CALUDE_bullet_speed_difference_l3612_361283

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in bullet speed when fired in the same direction as the horse versus the opposite direction -/
def speed_difference : ℝ := (bullet_speed + horse_speed) - (bullet_speed - horse_speed)

theorem bullet_speed_difference :
  speed_difference = 40 :=
by sorry

end NUMINAMATH_CALUDE_bullet_speed_difference_l3612_361283


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3612_361276

theorem arithmetic_geometric_sequence (a b : ℝ) : 
  (2 * a = 1 + b) →  -- arithmetic sequence condition
  (b^2 = a) →        -- geometric sequence condition
  (a ≠ b) →          -- given condition
  (a = 1/4) :=       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3612_361276


namespace NUMINAMATH_CALUDE_parabola_f_value_l3612_361280

/-- A parabola with equation y = dx^2 + ex + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.d * x^2 + p.e * x + p.f

theorem parabola_f_value (p : Parabola) :
  p.y_coord (-1) = -2 →  -- vertex condition
  p.y_coord 0 = -1.5 →   -- point condition
  p.f = -1.5 := by
sorry

end NUMINAMATH_CALUDE_parabola_f_value_l3612_361280


namespace NUMINAMATH_CALUDE_lilys_books_l3612_361202

theorem lilys_books (books_last_month : ℕ) : 
  books_last_month + 2 * books_last_month = 12 → books_last_month = 4 := by
  sorry

end NUMINAMATH_CALUDE_lilys_books_l3612_361202


namespace NUMINAMATH_CALUDE_percentage_of_girls_in_class_l3612_361255

theorem percentage_of_girls_in_class (B G : ℝ) :
  B > 0 →
  G > 0 →
  G + 0.5 * B = 1.5 * (0.5 * B) →
  (G / (B + G)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_girls_in_class_l3612_361255


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l3612_361244

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- State the theorem
theorem even_periodic_function_monotonicity (f : ℝ → ℝ) 
  (h_even : is_even f) (h_period : has_period f 2) :
  increasing_on f 0 1 ↔ decreasing_on f 3 4 :=
sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l3612_361244


namespace NUMINAMATH_CALUDE_stock_market_investment_l3612_361229

theorem stock_market_investment (initial_investment : ℝ) (h_positive : initial_investment > 0) :
  let first_year := initial_investment * 1.75
  let second_year := initial_investment * 1.225
  (first_year - second_year) / first_year = 0.3 := by
sorry

end NUMINAMATH_CALUDE_stock_market_investment_l3612_361229


namespace NUMINAMATH_CALUDE_base6_multiplication_addition_l3612_361296

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6, represented as a list of digits -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- The main theorem statement -/
theorem base6_multiplication_addition :
  let a := base6ToBase10 [1, 1, 1]  -- 111₆
  let b := 2
  let c := base6ToBase10 [2, 0, 2]  -- 202₆
  base10ToBase6 (a * b + c) = [4, 2, 4] := by
  sorry


end NUMINAMATH_CALUDE_base6_multiplication_addition_l3612_361296


namespace NUMINAMATH_CALUDE_product_of_digits_7891_base7_is_zero_l3612_361250

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 7 representation of 7891 is 0 -/
theorem product_of_digits_7891_base7_is_zero :
  productOfList (toBase7 7891) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_7891_base7_is_zero_l3612_361250


namespace NUMINAMATH_CALUDE_automobile_distance_l3612_361238

/-- 
Given an automobile that travels 2a/5 feet in r seconds, 
this theorem proves that it will travel 40a/r yards in 5 minutes 
if this rate is maintained.
-/
theorem automobile_distance (a r : ℝ) (hr : r > 0) : 
  let rate_feet_per_second := (2 * a / 5) / r
  let rate_yards_per_second := rate_feet_per_second / 3
  let time_in_seconds := 5 * 60
  rate_yards_per_second * time_in_seconds = 40 * a / r := by
  sorry

end NUMINAMATH_CALUDE_automobile_distance_l3612_361238


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3612_361270

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.width + field.length)

/-- Calculates the area of a rectangular field -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem: The area of a rectangular field with perimeter 120 meters and width 20 meters is 800 square meters -/
theorem rectangular_field_area :
  ∀ (field : RectangularField),
    field.width = 20 →
    perimeter field = 120 →
    area field = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3612_361270


namespace NUMINAMATH_CALUDE_sequence_integer_count_l3612_361242

def sequence_term (n : ℕ) : ℚ :=
  8820 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem sequence_integer_count :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ k < n, is_integer (sequence_term k)) ∧
    ¬is_integer (sequence_term n)) ∧
  (∀ m : ℕ, m > 0 →
    (∀ k < m, is_integer (sequence_term k)) →
    ¬is_integer (sequence_term m) →
    m = 3) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l3612_361242


namespace NUMINAMATH_CALUDE_age_sum_problem_l3612_361211

theorem age_sum_problem (twin1_age twin2_age youngest_age : ℕ) :
  twin1_age = twin2_age →
  twin1_age > youngest_age →
  youngest_age < 10 →
  twin1_age * twin2_age * youngest_age = 72 →
  twin1_age + twin2_age + youngest_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l3612_361211


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l3612_361274

theorem consecutive_squares_sum (x : ℕ) :
  x^2 + (x+1)^2 + (x+2)^2 = 2030 → x + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l3612_361274


namespace NUMINAMATH_CALUDE_f_properties_l3612_361260

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x ∈ Set.Icc (-3) (-2), f x ≤ -4) ∧
  (∀ x ∈ Set.Icc (-3) (-2), f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-3) (-2), f x = -4) ∧
  (∃ x ∈ Set.Icc (-3) (-2), f x = -7) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l3612_361260


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_cubed_minus_5_squared_l3612_361249

theorem least_prime_factor_of_5_cubed_minus_5_squared : 
  (Nat.minFac (5^3 - 5^2) = 2) := by sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_cubed_minus_5_squared_l3612_361249


namespace NUMINAMATH_CALUDE_tan_equation_solution_l3612_361259

open Set Real

-- Define the set of angles that satisfy the conditions
def solution_set : Set ℝ := {x | 0 ≤ x ∧ x < π ∧ tan (4 * x - π / 4) = 1}

-- State the theorem
theorem tan_equation_solution :
  solution_set = {π/8, 3*π/8, 5*π/8, 7*π/8} := by sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l3612_361259


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_closed_interval_l3612_361222

/-- Set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

/-- Set B defined in terms of x ∈ A -/
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 + 4*x + 3}

/-- The intersection of A and B is equal to the closed interval [-1, 1] -/
theorem A_intersect_B_eq_closed_interval :
  A ∩ B = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_closed_interval_l3612_361222


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l3612_361234

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l3612_361234


namespace NUMINAMATH_CALUDE_books_difference_alicia_ian_l3612_361235

/-- Represents a student in the book reading contest -/
structure Student where
  name : String
  booksRead : Nat

/-- Represents the book reading contest -/
structure BookReadingContest where
  students : Finset Student
  alicia : Student
  ian : Student
  aliciaMostBooks : ∀ s ∈ students, s.booksRead ≤ alicia.booksRead
  ianFewestBooks : ∀ s ∈ students, ian.booksRead ≤ s.booksRead
  aliciaInContest : alicia ∈ students
  ianInContest : ian ∈ students
  contestSize : students.card = 8
  aliciaBooksRead : alicia.booksRead = 8
  ianBooksRead : ian.booksRead = 1

/-- The difference in books read between Alicia and Ian is 7 -/
theorem books_difference_alicia_ian (contest : BookReadingContest) :
  contest.alicia.booksRead - contest.ian.booksRead = 7 := by
  sorry

end NUMINAMATH_CALUDE_books_difference_alicia_ian_l3612_361235


namespace NUMINAMATH_CALUDE_initial_trees_count_l3612_361281

/-- The number of walnut trees in the park after planting -/
def total_trees : ℕ := 55

/-- The number of walnut trees planted today -/
def planted_trees : ℕ := 33

/-- The initial number of walnut trees in the park -/
def initial_trees : ℕ := total_trees - planted_trees

theorem initial_trees_count : initial_trees = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_count_l3612_361281
