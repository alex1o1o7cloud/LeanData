import Mathlib

namespace NUMINAMATH_CALUDE_specific_parallelepiped_face_areas_l715_71587

/-- Represents a parallelepiped with given properties -/
structure Parallelepiped where
  h₁ : ℝ  -- Distance from first vertex to opposite face
  h₂ : ℝ  -- Distance from second vertex to opposite face
  h₃ : ℝ  -- Distance from third vertex to opposite face
  total_surface_area : ℝ  -- Total surface area of the parallelepiped

/-- The areas of the faces of the parallelepiped -/
def face_areas (p : Parallelepiped) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the areas of the faces for a specific parallelepiped -/
theorem specific_parallelepiped_face_areas :
  let p : Parallelepiped := {
    h₁ := 2,
    h₂ := 3,
    h₃ := 4,
    total_surface_area := 36
  }
  face_areas p = (108/13, 72/13, 54/13) := by sorry

end NUMINAMATH_CALUDE_specific_parallelepiped_face_areas_l715_71587


namespace NUMINAMATH_CALUDE_factorization_validity_l715_71518

theorem factorization_validity (x y : ℝ) :
  x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_validity_l715_71518


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l715_71520

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (b = c) →                  -- The triangle is isosceles
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (a + b + c = 22)           -- The perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 4 9 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l715_71520


namespace NUMINAMATH_CALUDE_gcd_of_160_200_360_l715_71524

theorem gcd_of_160_200_360 : Nat.gcd 160 (Nat.gcd 200 360) = 40 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_160_200_360_l715_71524


namespace NUMINAMATH_CALUDE_composite_product_division_l715_71583

def first_six_composite_product : ℕ := 4 * 6 * 8 * 9 * 10 * 12
def next_six_composite_product : ℕ := 14 * 15 * 16 * 18 * 20 * 21

theorem composite_product_division :
  (first_six_composite_product : ℚ) / next_six_composite_product = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_division_l715_71583


namespace NUMINAMATH_CALUDE_college_cost_calculation_l715_71548

/-- The total cost of Sabina's first year of college -/
def total_cost : ℝ := 30000

/-- Sabina's savings -/
def savings : ℝ := 10000

/-- The percentage of the remainder covered by the grant -/
def grant_percentage : ℝ := 0.40

/-- The amount of the loan Sabina needs -/
def loan_amount : ℝ := 12000

/-- Theorem stating that the total cost is correct given the conditions -/
theorem college_cost_calculation :
  total_cost = savings + grant_percentage * (total_cost - savings) + loan_amount := by
  sorry

end NUMINAMATH_CALUDE_college_cost_calculation_l715_71548


namespace NUMINAMATH_CALUDE_unanswered_questions_l715_71539

theorem unanswered_questions 
  (total_questions : ℕ) 
  (time_spent : ℕ) 
  (time_per_question : ℕ) 
  (h1 : total_questions = 100)
  (h2 : time_spent = 120) -- 2 hours in minutes
  (h3 : time_per_question = 2) :
  total_questions - (time_spent / time_per_question) = 40 :=
by sorry

end NUMINAMATH_CALUDE_unanswered_questions_l715_71539


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l715_71593

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_sum : a 2 + a 4 = 3)
  (h_prod : a 3 * a 5 = 2) :
  q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l715_71593


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l715_71522

/-- Given an arithmetic sequence of 20 terms with first term 2 and last term 59,
    prove that the 5th term is 14. -/
theorem fifth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  a 0 = 2 →                            -- first term is 2
  a 19 = 59 →                          -- last term (20th term) is 59
  a 4 = 14 :=                          -- 5th term (index 4) is 14
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l715_71522


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l715_71576

/-- Represents the loan conditions and proves the principal amount --/
theorem loan_principal_calculation (initial_fee_rate : ℝ) (weeks : ℕ) (total_fee : ℝ) (principal : ℝ) : 
  initial_fee_rate = 0.05 →
  weeks = 2 →
  total_fee = 15 →
  (initial_fee_rate * principal) + (2 * initial_fee_rate * principal) = total_fee →
  principal = 100 := by
  sorry

#check loan_principal_calculation

end NUMINAMATH_CALUDE_loan_principal_calculation_l715_71576


namespace NUMINAMATH_CALUDE_sum_exterior_angles_regular_decagon_l715_71515

/-- A regular decagon is a polygon with 10 sides -/
def RegularDecagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def SumExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of exterior angles of a regular decagon is 360° -/
theorem sum_exterior_angles_regular_decagon :
  SumExteriorAngles RegularDecagon = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_regular_decagon_l715_71515


namespace NUMINAMATH_CALUDE_baker_revenue_difference_l715_71590

/-- Baker's sales and pricing information --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculate the difference between daily average and today's revenue --/
def revenue_difference (sales : BakerSales) : ℕ :=
  let usual_revenue := sales.usual_pastries * sales.pastry_price + sales.usual_bread * sales.bread_price
  let today_revenue := sales.today_pastries * sales.pastry_price + sales.today_bread * sales.bread_price
  today_revenue - usual_revenue

/-- Theorem stating the revenue difference for the given sales information --/
theorem baker_revenue_difference :
  revenue_difference ⟨20, 10, 14, 25, 2, 4⟩ = 48 := by
  sorry

end NUMINAMATH_CALUDE_baker_revenue_difference_l715_71590


namespace NUMINAMATH_CALUDE_jimin_candies_l715_71541

/-- The number of candies Jimin gave to Yuna -/
def candies_given : ℕ := 25

/-- The number of candies left over -/
def candies_left : ℕ := 13

/-- The total number of candies Jimin had at the start -/
def total_candies : ℕ := candies_given + candies_left

theorem jimin_candies : total_candies = 38 := by
  sorry

end NUMINAMATH_CALUDE_jimin_candies_l715_71541


namespace NUMINAMATH_CALUDE_polynomial_factorization_l715_71504

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (a*b^2 + b*c^2 + c*a^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l715_71504


namespace NUMINAMATH_CALUDE_linear_function_with_constraints_l715_71547

def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

def PassesThroughPoint (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def IntersectsPositiveXAxis (f : ℝ → ℝ) (x : ℝ) : Prop :=
  x > 0 ∧ f x = 0

def IntersectsPositiveYAxis (f : ℝ → ℝ) (y : ℝ) : Prop :=
  y > 0 ∧ f 0 = y

theorem linear_function_with_constraints (f : ℝ → ℝ) 
    (h_linear : LinearFunction f)
    (h_point : PassesThroughPoint f 3 2)
    (h_x_intersect : ∃ a : ℝ, IntersectsPositiveXAxis f a)
    (h_y_intersect : ∃ b : ℝ, IntersectsPositiveYAxis f b)
    (h_sum : ∃ a b : ℝ, IntersectsPositiveXAxis f a ∧ 
                        IntersectsPositiveYAxis f b ∧ 
                        a + b = 12) :
  (∀ x : ℝ, f x = -2 * x + 8) ∨ (∀ x : ℝ, f x = -1/3 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_with_constraints_l715_71547


namespace NUMINAMATH_CALUDE_race_head_start_l715_71535

theorem race_head_start (L : ℝ) (vₐ vᵦ : ℝ) (h : vₐ = (17 / 14) * vᵦ) :
  let x := (3 / 17) * L
  L / vₐ = (L - x) / vᵦ :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l715_71535


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l715_71579

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 1 - 9
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l715_71579


namespace NUMINAMATH_CALUDE_distance_between_points_l715_71500

/-- The distance between two points (3, 0) and (7, 7) on a Cartesian coordinate plane is √65. -/
theorem distance_between_points : Real.sqrt 65 = Real.sqrt ((7 - 3)^2 + (7 - 0)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l715_71500


namespace NUMINAMATH_CALUDE_roots_equation_result_l715_71513

theorem roots_equation_result (γ δ : ℝ) : 
  γ^2 - 3*γ + 1 = 0 → δ^2 - 3*δ + 1 = 0 → 8*γ^3 + 15*δ^2 = 179 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_result_l715_71513


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l715_71552

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : IsIncreasingGeometricSequence a)
  (h_sum : a 1 + a 4 = 9)
  (h_prod : a 2 * a 3 = 8) :
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l715_71552


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l715_71594

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- For any geometric sequence, the sum of squares of the first and third terms is greater than or equal to twice the square of the second term. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : IsGeometricSequence a) :
    a 1 ^ 2 + a 3 ^ 2 ≥ 2 * (a 2 ^ 2) :=
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_inequality_l715_71594


namespace NUMINAMATH_CALUDE_mixed_numbers_sum_range_l715_71521

theorem mixed_numbers_sum_range : 
  let a : ℚ := 3 + 1 / 9
  let b : ℚ := 4 + 1 / 3
  let c : ℚ := 6 + 1 / 21
  let sum : ℚ := a + b + c
  13.5 < sum ∧ sum < 14 := by
sorry

end NUMINAMATH_CALUDE_mixed_numbers_sum_range_l715_71521


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l715_71526

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l715_71526


namespace NUMINAMATH_CALUDE_fraction_of_number_l715_71519

theorem fraction_of_number (N : ℝ) : (0.4 * N = 204) → ((1/4) * (1/3) * (2/5) * N = 17) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l715_71519


namespace NUMINAMATH_CALUDE_intersection_sum_l715_71507

theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 7 ∧ y = 4 * x + b → x = 8 ∧ y = 11) → 
  b + m = -20.5 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l715_71507


namespace NUMINAMATH_CALUDE_speedster_convertibles_l715_71573

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) 
  (h1 : speedsters = (2 * total) / 3)
  (h2 : convertibles = (4 * speedsters) / 5)
  (h3 : total - speedsters = 60) :
  convertibles = 96 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l715_71573


namespace NUMINAMATH_CALUDE_total_product_weight_is_correct_l715_71554

/-- Represents a chemical element or compound -/
structure Chemical where
  formula : String
  molarMass : Float

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List (Chemical × Float)
  products : List (Chemical × Float)

def CaCO3 : Chemical := ⟨"CaCO3", 100.09⟩
def CaO : Chemical := ⟨"CaO", 56.08⟩
def CO2 : Chemical := ⟨"CO2", 44.01⟩
def HCl : Chemical := ⟨"HCl", 36.46⟩
def CaCl2 : Chemical := ⟨"CaCl2", 110.98⟩
def H2O : Chemical := ⟨"H2O", 18.02⟩

def reaction1 : Reaction := ⟨[(CaCO3, 1)], [(CaO, 1), (CO2, 1)]⟩
def reaction2 : Reaction := ⟨[(HCl, 2), (CaCO3, 1)], [(CaCl2, 1), (CO2, 1), (H2O, 1)]⟩

def initialCaCO3 : Float := 8
def initialHCl : Float := 12

/-- Calculates the total weight of products from both reactions -/
def totalProductWeight (r1 : Reaction) (r2 : Reaction) (initCaCO3 : Float) (initHCl : Float) : Float :=
  sorry

theorem total_product_weight_is_correct :
  totalProductWeight reaction1 reaction2 initialCaCO3 initialHCl = 800.72 := by sorry

end NUMINAMATH_CALUDE_total_product_weight_is_correct_l715_71554


namespace NUMINAMATH_CALUDE_sandbox_ratio_l715_71546

theorem sandbox_ratio (L W H k : ℝ) : 
  L > 0 → W > 0 → H > 0 → k > 0 →
  L * W * H = 10 →
  (k * L) * (k * W) * (k * H) = 80 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_sandbox_ratio_l715_71546


namespace NUMINAMATH_CALUDE_max_basketballs_proof_l715_71523

/-- The maximum number of basketballs that can be purchased given the constraints -/
def max_basketballs : ℕ := 26

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 40

/-- The cost of each basketball in dollars -/
def basketball_cost : ℕ := 80

/-- The cost of each soccer ball in dollars -/
def soccer_ball_cost : ℕ := 50

/-- The total budget in dollars -/
def total_budget : ℕ := 2800

theorem max_basketballs_proof :
  (∀ x : ℕ, 
    x ≤ total_balls ∧ 
    (basketball_cost * x + soccer_ball_cost * (total_balls - x) ≤ total_budget) →
    x ≤ max_basketballs) ∧
  (basketball_cost * max_basketballs + soccer_ball_cost * (total_balls - max_basketballs) ≤ total_budget) :=
sorry

end NUMINAMATH_CALUDE_max_basketballs_proof_l715_71523


namespace NUMINAMATH_CALUDE_speed_against_current_l715_71553

def distance : ℝ := 30
def time_downstream : ℝ := 2
def time_upstream : ℝ := 3

def speed_downstream (v_m v_c : ℝ) : ℝ := v_m + v_c
def speed_upstream (v_m v_c : ℝ) : ℝ := v_m - v_c

theorem speed_against_current :
  ∃ (v_m v_c : ℝ),
    distance = speed_downstream v_m v_c * time_downstream ∧
    distance = speed_upstream v_m v_c * time_upstream ∧
    speed_upstream v_m v_c = 10 :=
by sorry

end NUMINAMATH_CALUDE_speed_against_current_l715_71553


namespace NUMINAMATH_CALUDE_annas_gold_cost_per_gram_l715_71582

/-- Calculates the cost per gram of Anna's gold -/
theorem annas_gold_cost_per_gram 
  (gary_gold : ℝ) 
  (gary_cost_per_gram : ℝ) 
  (anna_gold : ℝ) 
  (total_cost : ℝ) 
  (h1 : gary_gold = 30)
  (h2 : gary_cost_per_gram = 15)
  (h3 : anna_gold = 50)
  (h4 : total_cost = 1450)
  (h5 : total_cost = gary_gold * gary_cost_per_gram + anna_gold * (total_cost - gary_gold * gary_cost_per_gram) / anna_gold) :
  (total_cost - gary_gold * gary_cost_per_gram) / anna_gold = 20 := by
  sorry

#check annas_gold_cost_per_gram

end NUMINAMATH_CALUDE_annas_gold_cost_per_gram_l715_71582


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l715_71588

/-- The function f(x) = a^(x-2) - 1 always passes through the point (2, 0) for any a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 1
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l715_71588


namespace NUMINAMATH_CALUDE_amandas_quiz_average_l715_71556

theorem amandas_quiz_average :
  ∀ (num_quizzes : ℕ) (final_quiz_score : ℝ) (required_average : ℝ),
    num_quizzes = 4 →
    final_quiz_score = 97 →
    required_average = 93 →
    ∃ (current_average : ℝ),
      current_average = 92 ∧
      (num_quizzes : ℝ) * current_average + final_quiz_score = (num_quizzes + 1 : ℝ) * required_average :=
by
  sorry

end NUMINAMATH_CALUDE_amandas_quiz_average_l715_71556


namespace NUMINAMATH_CALUDE_card_A_total_percent_decrease_l715_71558

def card_A_initial_value : ℝ := 150
def card_A_decrease_year1 : ℝ := 0.20
def card_A_decrease_year2 : ℝ := 0.30
def card_A_decrease_year3 : ℝ := 0.15

def card_A_value_after_three_years : ℝ :=
  card_A_initial_value * (1 - card_A_decrease_year1) * (1 - card_A_decrease_year2) * (1 - card_A_decrease_year3)

theorem card_A_total_percent_decrease :
  (card_A_initial_value - card_A_value_after_three_years) / card_A_initial_value = 0.524 := by
  sorry

end NUMINAMATH_CALUDE_card_A_total_percent_decrease_l715_71558


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l715_71505

def Point := ℝ × ℝ

def symmetric_origin (p1 p2 : Point) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

def symmetric_y_axis (p1 p2 : Point) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetric_point_coordinates :
  ∀ (P P1 P2 : Point),
    symmetric_origin P1 P →
    P1 = (-2, 3) →
    symmetric_y_axis P2 P →
    P2 = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l715_71505


namespace NUMINAMATH_CALUDE_smaller_number_proof_l715_71578

theorem smaller_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 1860) (h3 : 0.075 * x = 0.125 * y) : y = 2790 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l715_71578


namespace NUMINAMATH_CALUDE_total_height_calculation_l715_71562

-- Define the heights in inches
def sculpture_height_inches : ℚ := 34
def base_height_inches : ℚ := 2

-- Define the conversion factor from inches to centimeters
def inches_to_cm : ℚ := 2.54

-- Define the total height in centimeters
def total_height_cm : ℚ := (sculpture_height_inches + base_height_inches) * inches_to_cm

-- Theorem statement
theorem total_height_calculation :
  total_height_cm = 91.44 := by sorry

end NUMINAMATH_CALUDE_total_height_calculation_l715_71562


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l715_71584

theorem cubic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c)^3 - (a^3 + b^3 + c^3) > (a + b) * (b + c) * (a + c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l715_71584


namespace NUMINAMATH_CALUDE_cruz_marbles_l715_71545

/-- Proof that Cruz has 8 marbles given the conditions of the problem -/
theorem cruz_marbles :
  ∀ (atticus jensen cruz : ℕ),
  3 * (atticus + jensen + cruz) = 60 →
  atticus = jensen / 2 →
  atticus = 4 →
  cruz = 8 := by
sorry

end NUMINAMATH_CALUDE_cruz_marbles_l715_71545


namespace NUMINAMATH_CALUDE_factor_expression_l715_71565

theorem factor_expression (b : ℝ) : 43 * b^2 + 129 * b = 43 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l715_71565


namespace NUMINAMATH_CALUDE_dog_catch_ball_time_l715_71528

/-- The time it takes for a dog to catch up to a thrown ball -/
theorem dog_catch_ball_time (ball_speed : ℝ) (ball_time : ℝ) (dog_speed : ℝ) :
  ball_speed = 20 →
  ball_time = 8 →
  dog_speed = 5 →
  (ball_speed * ball_time) / dog_speed = 32 := by
  sorry

#check dog_catch_ball_time

end NUMINAMATH_CALUDE_dog_catch_ball_time_l715_71528


namespace NUMINAMATH_CALUDE_science_club_election_l715_71538

theorem science_club_election (total_candidates : Nat) (past_officers : Nat) (positions : Nat) :
  total_candidates = 20 →
  past_officers = 8 →
  positions = 6 →
  (Nat.choose total_candidates positions -
   (Nat.choose (total_candidates - past_officers) positions +
    Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1))) = 31500 :=
by sorry

end NUMINAMATH_CALUDE_science_club_election_l715_71538


namespace NUMINAMATH_CALUDE_tenth_fibonacci_is_55_l715_71543

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem tenth_fibonacci_is_55 : fibonacci 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_tenth_fibonacci_is_55_l715_71543


namespace NUMINAMATH_CALUDE_fahrenheit_from_kelvin_l715_71561

theorem fahrenheit_from_kelvin (K F C : ℝ) : 
  K = 300 → 
  C = (5/9) * (F - 32) → 
  C = K - 273 → 
  F = 80.6 := by sorry

end NUMINAMATH_CALUDE_fahrenheit_from_kelvin_l715_71561


namespace NUMINAMATH_CALUDE_joes_first_lift_weight_l715_71550

theorem joes_first_lift_weight (total_weight first_lift second_lift : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift + second_lift = total_weight)
  (h3 : 2 * first_lift = second_lift + 300) :
  first_lift = 400 := by
  sorry

end NUMINAMATH_CALUDE_joes_first_lift_weight_l715_71550


namespace NUMINAMATH_CALUDE_rope_and_well_depth_l715_71525

/-- Given a rope of length L and a well of depth H, prove that if L/2 + 9 = H and L/3 + 2 = H, then L = 42 and H = 30. -/
theorem rope_and_well_depth (L H : ℝ) 
  (h1 : L/2 + 9 = H) 
  (h2 : L/3 + 2 = H) : 
  L = 42 ∧ H = 30 := by
sorry

end NUMINAMATH_CALUDE_rope_and_well_depth_l715_71525


namespace NUMINAMATH_CALUDE_salesperson_allocation_l715_71580

/-- Represents the problem of determining the number of salespersons to send to a branch office --/
theorem salesperson_allocation
  (total_salespersons : ℕ)
  (initial_avg_income : ℝ)
  (hq_income_increase : ℝ)
  (branch_income_factor : ℝ)
  (h_total : total_salespersons = 100)
  (h_hq_increase : hq_income_increase = 0.2)
  (h_branch_factor : branch_income_factor = 3.5)
  (x : ℕ) :
  (((total_salespersons - x) * (1 + hq_income_increase) * initial_avg_income ≥ 
    total_salespersons * initial_avg_income) ∧
   (x * branch_income_factor * initial_avg_income ≥ 
    0.5 * total_salespersons * initial_avg_income)) →
  (x = 15 ∨ x = 16) :=
by sorry

end NUMINAMATH_CALUDE_salesperson_allocation_l715_71580


namespace NUMINAMATH_CALUDE_proposition_truth_l715_71568

theorem proposition_truth (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : p ∨ q) : 
  (¬p) ∨ (¬q) := by
sorry

end NUMINAMATH_CALUDE_proposition_truth_l715_71568


namespace NUMINAMATH_CALUDE_point_c_coordinates_l715_71555

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The area of a triangle given three points -/
def triangleArea (a b c : Point2D) : ℝ := sorry

/-- Theorem: Given the conditions, point C has coordinates (0,4) or (0,-4) -/
theorem point_c_coordinates :
  let a : Point2D := ⟨-2, 0⟩
  let b : Point2D := ⟨3, 0⟩
  ∀ c : Point2D,
    c.x = 0 →  -- C lies on the y-axis
    triangleArea a b c = 10 →
    (c.y = 4 ∨ c.y = -4) :=
by sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l715_71555


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l715_71530

/-- Proves that the profit from selling a cricket bat is approximately $215.29 --/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) (h1 : selling_price = 850) (h2 : profit_percentage = 33.85826771653544) :
  ∃ (profit : ℝ), abs (profit - 215.29) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l715_71530


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l715_71502

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (2025 * x))^4 + (Real.cos (2016 * x))^2019 * (Real.cos (2025 * x))^2018 = 1 ↔ 
  (∃ n : ℤ, x = π / 4050 + π * n / 2025) ∨ (∃ k : ℤ, x = π * k / 9) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l715_71502


namespace NUMINAMATH_CALUDE_system_of_equations_solution_fractional_equation_solution_l715_71529

-- Problem 1: System of equations
theorem system_of_equations_solution :
  ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 :=
sorry

-- Problem 2: Fractional equation
theorem fractional_equation_solution :
  ∃! y : ℝ, y ≠ 1 ∧ 3 / (1 - y) = y / (y - 1) - 5 :=
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_fractional_equation_solution_l715_71529


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l715_71591

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ (b : ℝ), (1 + a * Complex.I) / (2 - Complex.I) = b * Complex.I) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l715_71591


namespace NUMINAMATH_CALUDE_invisible_square_exists_l715_71536

/-- A point with integer coordinates is invisible if the gcd of its coordinates is greater than 1 -/
def invisible (p q : ℤ) : Prop := Nat.gcd p.natAbs q.natAbs > 1

/-- There exists a square with side length n*k where all integer coordinate points are invisible -/
theorem invisible_square_exists (n : ℕ) : ∃ k : ℕ, k ≥ 2 ∧ 
  ∀ p q : ℤ, 0 ≤ p ∧ p ≤ n * k ∧ 0 ≤ q ∧ q ≤ n * k → invisible p q :=
sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l715_71536


namespace NUMINAMATH_CALUDE_point_range_theorem_l715_71511

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3 * x - 2 * y + a = 0

-- Define the condition for two points being on the same side of the line
def same_side (x1 y1 x2 y2 a : ℝ) : Prop :=
  (3 * x1 - 2 * y1 + a) * (3 * x2 - 2 * y2 + a) > 0

-- Theorem statement
theorem point_range_theorem (a : ℝ) :
  line_equation 3 (-1) a ∧ line_equation (-4) (-3) a ∧ same_side 3 (-1) (-4) (-3) a
  ↔ a < -11 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_point_range_theorem_l715_71511


namespace NUMINAMATH_CALUDE_weighted_average_calculation_l715_71564

/-- Calculates the weighted average of exam scores for a class -/
theorem weighted_average_calculation (total_students : ℕ) 
  (math_perfect_scores math_zero_scores : ℕ)
  (science_perfect_scores science_zero_scores : ℕ)
  (math_average_rest science_average_rest : ℚ)
  (math_weight science_weight : ℚ) :
  total_students = 30 →
  math_perfect_scores = 3 →
  math_zero_scores = 4 →
  math_average_rest = 50 →
  science_perfect_scores = 2 →
  science_zero_scores = 5 →
  science_average_rest = 60 →
  math_weight = 2/5 →
  science_weight = 3/5 →
  (((math_perfect_scores * 100 + 
    (total_students - math_perfect_scores - math_zero_scores) * math_average_rest) * math_weight +
   (science_perfect_scores * 100 + 
    (total_students - science_perfect_scores - science_zero_scores) * science_average_rest) * science_weight) / total_students) = 1528/30 :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_calculation_l715_71564


namespace NUMINAMATH_CALUDE_triangle_problem_l715_71549

open Real

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / sin A = b / sin B ∧ b / sin B = c / sin C)
  (h_condition : Real.sqrt 3 * b * sin C = c * cos B + c) : 
  B = π / 3 ∧ 
  (b^2 = a * c → 2 / tan A + 1 / tan C = 2 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l715_71549


namespace NUMINAMATH_CALUDE_spider_journey_l715_71560

theorem spider_journey (r : ℝ) (third_leg : ℝ) (h1 : r = 75) (h2 : third_leg = 110) : 
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - third_leg^2)
  diameter + second_leg + third_leg = 362 := by
sorry

end NUMINAMATH_CALUDE_spider_journey_l715_71560


namespace NUMINAMATH_CALUDE_walking_rate_l715_71551

/-- Given a distance of 4 miles and a time of 1.25 hours, the rate of travel is 3.2 miles per hour -/
theorem walking_rate (distance : ℝ) (time : ℝ) (rate : ℝ) 
    (h1 : distance = 4)
    (h2 : time = 1.25)
    (h3 : rate = distance / time) : 
  rate = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_l715_71551


namespace NUMINAMATH_CALUDE_keith_attended_four_games_l715_71577

/-- The number of football games Keith attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Theorem: Keith attended 4 football games -/
theorem keith_attended_four_games (total_games missed_games : ℕ) 
  (h1 : total_games = 8)
  (h2 : missed_games = 4) : 
  games_attended total_games missed_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_keith_attended_four_games_l715_71577


namespace NUMINAMATH_CALUDE_rohan_sudhir_profit_difference_l715_71585

/-- Represents an investor in the business -/
structure Investor where
  name : String
  amount : ℕ
  months : ℕ

/-- Calculates the investment-time product for an investor -/
def investmentTime (i : Investor) : ℕ := i.amount * i.months

/-- Calculates the share of profit for an investor -/
def profitShare (i : Investor) (totalInvestmentTime totalProfit : ℕ) : ℚ :=
  (investmentTime i : ℚ) / totalInvestmentTime * totalProfit

theorem rohan_sudhir_profit_difference 
  (suresh : Investor)
  (rohan : Investor)
  (sudhir : Investor)
  (priya : Investor)
  (akash : Investor)
  (totalProfit : ℕ) :
  suresh.name = "Suresh" ∧ suresh.amount = 18000 ∧ suresh.months = 12 ∧
  rohan.name = "Rohan" ∧ rohan.amount = 12000 ∧ rohan.months = 9 ∧
  sudhir.name = "Sudhir" ∧ sudhir.amount = 9000 ∧ sudhir.months = 8 ∧
  priya.name = "Priya" ∧ priya.amount = 15000 ∧ priya.months = 6 ∧
  akash.name = "Akash" ∧ akash.amount = 10000 ∧ akash.months = 6 ∧
  totalProfit = 5948 →
  let totalInvestmentTime := investmentTime suresh + investmentTime rohan + 
                             investmentTime sudhir + investmentTime priya + 
                             investmentTime akash
  (profitShare rohan totalInvestmentTime totalProfit - 
   profitShare sudhir totalInvestmentTime totalProfit).num = 393 :=
by sorry

end NUMINAMATH_CALUDE_rohan_sudhir_profit_difference_l715_71585


namespace NUMINAMATH_CALUDE_oldest_sibling_age_difference_l715_71510

theorem oldest_sibling_age_difference (siblings : Fin 4 → ℝ) 
  (avg_age : (siblings 0 + siblings 1 + siblings 2 + siblings 3) / 4 = 30)
  (youngest_age : siblings 3 = 25.75)
  : ∃ i : Fin 4, siblings i - siblings 3 ≥ 17 :=
by
  sorry

end NUMINAMATH_CALUDE_oldest_sibling_age_difference_l715_71510


namespace NUMINAMATH_CALUDE_smallest_non_trivial_divisor_of_product_l715_71569

def product_of_even_integers (n : ℕ) : ℕ :=
  (List.range ((n + 1) / 2)).foldl (λ acc i => acc * (2 * (i + 1))) 1

theorem smallest_non_trivial_divisor_of_product (n : ℕ) (h : n = 134) :
  ∃ (d : ℕ), d > 1 ∧ d ∣ product_of_even_integers n ∧
  ∀ (k : ℕ), 1 < k → k < d → ¬(k ∣ product_of_even_integers n) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_non_trivial_divisor_of_product_l715_71569


namespace NUMINAMATH_CALUDE_difference_of_squares_65_55_l715_71544

theorem difference_of_squares_65_55 : 65^2 - 55^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_55_l715_71544


namespace NUMINAMATH_CALUDE_inscribed_triangle_tangent_theorem_l715_71559

/-- A parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop

/-- A triangle in the xy-plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Checks if a line is tangent to a parabola -/
def is_tangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (parabola : Parabola) : Prop :=
  sorry

/-- Main theorem: If two sides of an inscribed triangle are tangent to a parabola, the third side is also tangent -/
theorem inscribed_triangle_tangent_theorem
  (p q : ℝ)
  (parabola1 : Parabola)
  (parabola2 : Parabola)
  (triangle : Triangle)
  (h1 : p > 0)
  (h2 : q > 0)
  (h3 : parabola1.equation = fun x y ↦ y^2 = 2*p*x)
  (h4 : parabola2.equation = fun x y ↦ x^2 = 2*q*y)
  (h5 : ∀ (x y : ℝ), parabola1.equation x y → (x = triangle.A.1 ∧ y = triangle.A.2) ∨ 
                                               (x = triangle.B.1 ∧ y = triangle.B.2) ∨ 
                                               (x = triangle.C.1 ∧ y = triangle.C.2))
  (h6 : is_tangent (fun A B ↦ ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
         (triangle.A.1 + t * (triangle.B.1 - triangle.A.1) = A.1) ∧
         (triangle.A.2 + t * (triangle.B.2 - triangle.A.2) = A.2)) parabola2)
  (h7 : is_tangent (fun B C ↦ ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
         (triangle.B.1 + t * (triangle.C.1 - triangle.B.1) = B.1) ∧
         (triangle.B.2 + t * (triangle.C.2 - triangle.B.2) = B.2)) parabola2)
  : is_tangent (fun A C ↦ ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
         (triangle.A.1 + t * (triangle.C.1 - triangle.A.1) = A.1) ∧
         (triangle.A.2 + t * (triangle.C.2 - triangle.A.2) = A.2)) parabola2 :=
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_tangent_theorem_l715_71559


namespace NUMINAMATH_CALUDE_square_of_z_l715_71595

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := 2 + 5 * i

-- Theorem statement
theorem square_of_z : z^2 = -21 + 20 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_z_l715_71595


namespace NUMINAMATH_CALUDE_expected_rain_total_l715_71517

/-- The number of days in the weather forecast. -/
def num_days : ℕ := 5

/-- The probability of a sunny day with no rain. -/
def prob_sun : ℝ := 0.4

/-- The probability of a day with 4 inches of rain. -/
def prob_rain_4 : ℝ := 0.25

/-- The probability of a day with 10 inches of rain. -/
def prob_rain_10 : ℝ := 0.35

/-- The amount of rain on a sunny day. -/
def rain_sun : ℝ := 0

/-- The amount of rain on a day with 4 inches of rain. -/
def rain_4 : ℝ := 4

/-- The amount of rain on a day with 10 inches of rain. -/
def rain_10 : ℝ := 10

/-- The expected value of rain for a single day. -/
def expected_rain_day : ℝ :=
  prob_sun * rain_sun + prob_rain_4 * rain_4 + prob_rain_10 * rain_10

/-- Theorem: The expected value of the total number of inches of rain for 5 days is 22.5 inches. -/
theorem expected_rain_total : num_days * expected_rain_day = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_rain_total_l715_71517


namespace NUMINAMATH_CALUDE_hazel_caught_24_salmons_l715_71532

/-- Represents the number of salmons caught by Hazel and her father -/
structure FishingTrip where
  total : ℕ
  father : ℕ

/-- Calculates the number of salmons Hazel caught -/
def hazel_catch (trip : FishingTrip) : ℕ :=
  trip.total - trip.father

/-- Theorem: Given the conditions of the fishing trip, prove that Hazel caught 24 salmons -/
theorem hazel_caught_24_salmons (trip : FishingTrip)
  (h1 : trip.total = 51)
  (h2 : trip.father = 27) :
  hazel_catch trip = 24 := by
sorry

end NUMINAMATH_CALUDE_hazel_caught_24_salmons_l715_71532


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_halves_l715_71516

theorem infinite_solutions_iff_c_eq_five_halves (c : ℚ) :
  (∀ y : ℚ, 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_c_eq_five_halves_l715_71516


namespace NUMINAMATH_CALUDE_cubic_factorization_l715_71586

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x + 3)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l715_71586


namespace NUMINAMATH_CALUDE_house_deal_profit_l715_71592

theorem house_deal_profit (house_worth : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  house_worth = 10000 ∧ profit_percent = 10 ∧ loss_percent = 10 →
  let first_sale := house_worth * (1 + profit_percent / 100)
  let second_sale := first_sale * (1 - loss_percent / 100)
  first_sale - second_sale = 1100 := by
  sorry

end NUMINAMATH_CALUDE_house_deal_profit_l715_71592


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l715_71567

def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l715_71567


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l715_71570

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π/4) = 2) : 
  Real.tan x / Real.tan (2*x) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l715_71570


namespace NUMINAMATH_CALUDE_processing_block_performs_assignment_calculation_l715_71509

-- Define the types of program blocks
inductive ProgramBlock
  | Terminal
  | InputOutput
  | Processing
  | Decision

-- Define the functions that a block can perform
inductive BlockFunction
  | StartStop
  | InformationIO
  | AssignmentCalculation
  | ConditionCheck

-- Define a function that maps a block to its primary function
def blockPrimaryFunction : ProgramBlock → BlockFunction
  | ProgramBlock.Terminal => BlockFunction.StartStop
  | ProgramBlock.InputOutput => BlockFunction.InformationIO
  | ProgramBlock.Processing => BlockFunction.AssignmentCalculation
  | ProgramBlock.Decision => BlockFunction.ConditionCheck

-- Theorem statement
theorem processing_block_performs_assignment_calculation :
  ∀ (block : ProgramBlock),
    blockPrimaryFunction block = BlockFunction.AssignmentCalculation
    ↔ block = ProgramBlock.Processing :=
by sorry

end NUMINAMATH_CALUDE_processing_block_performs_assignment_calculation_l715_71509


namespace NUMINAMATH_CALUDE_max_remainder_dividend_l715_71599

theorem max_remainder_dividend (divisor quotient : ℕ) (h1 : divisor = 8) (h2 : quotient = 10) : 
  quotient * divisor + (divisor - 1) = 87 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_dividend_l715_71599


namespace NUMINAMATH_CALUDE_shift_proportional_function_l715_71575

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Shifts a linear function vertically by a given amount -/
def verticalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + shift }

theorem shift_proportional_function :
  let f : LinearFunction := { m := -2, b := 0 }
  let shifted_f := verticalShift f 3
  shifted_f = { m := -2, b := 3 } := by
  sorry

end NUMINAMATH_CALUDE_shift_proportional_function_l715_71575


namespace NUMINAMATH_CALUDE_not_parabola_l715_71531

/-- The equation x² + y²cos(θ) = 1, where θ is any real number, cannot represent a parabola -/
theorem not_parabola (θ : ℝ) : 
  ¬ (∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    (x^2 + y^2 * Real.cos θ = 1) ↔ (a*x^2 + b*x*y + c*y^2 + d*x + e*y = 1 ∧ b^2 = 4*a*c)) :=
by sorry

end NUMINAMATH_CALUDE_not_parabola_l715_71531


namespace NUMINAMATH_CALUDE_sibling_pair_implies_a_gt_one_l715_71533

/-- A point pair (x₁, y₁) and (x₂, y₂) is a "sibling point pair" for a function f
    if they both lie on the graph of f and are symmetric about the origin. -/
def is_sibling_point_pair (f : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  f x₁ = y₁ ∧ f x₂ = y₂ ∧ x₁ = -x₂ ∧ y₁ = -y₂

/-- The function f(x) = a^x - x - a has only one sibling point pair. -/
def has_unique_sibling_pair (a : ℝ) : Prop :=
  ∃! (x₁ y₁ x₂ y₂ : ℝ), is_sibling_point_pair (fun x => a^x - x - a) x₁ y₁ x₂ y₂

theorem sibling_pair_implies_a_gt_one (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) 
    (h₃ : has_unique_sibling_pair a) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_sibling_pair_implies_a_gt_one_l715_71533


namespace NUMINAMATH_CALUDE_stamp_price_l715_71503

theorem stamp_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 6 → purchase_price = (1 / 5) * original_price → original_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_stamp_price_l715_71503


namespace NUMINAMATH_CALUDE_car_speed_proof_l715_71571

/-- Proves that a car covering 400 meters in 12 seconds has a speed of 120 kilometers per hour -/
theorem car_speed_proof (distance : ℝ) (time : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) : 
  distance = 400 ∧ time = 12 ∧ speed_mps = distance / time ∧ speed_kmph = speed_mps * 3.6 →
  speed_kmph = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l715_71571


namespace NUMINAMATH_CALUDE_base6_addition_problem_l715_71506

/-- Converts a base 6 number to its decimal representation -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Checks if the given digit satisfies the base 6 addition problem -/
def satisfiesAdditionProblem (digit : Nat) : Prop :=
  let num1 := base6ToDecimal [4, 3, 2, digit]
  let num2 := base6ToDecimal [digit, 5, 1]
  let num3 := base6ToDecimal [digit, 3]
  let sum := base6ToDecimal [5, 3, digit, 0]
  num1 + num2 + num3 = sum

theorem base6_addition_problem :
  ∃! (digit : Nat), digit < 6 ∧ satisfiesAdditionProblem digit :=
sorry

end NUMINAMATH_CALUDE_base6_addition_problem_l715_71506


namespace NUMINAMATH_CALUDE_decimal_difference_l715_71512

def repeating_decimal : ℚ := 9 / 11
def terminating_decimal : ℚ := 81 / 100

theorem decimal_difference : repeating_decimal - terminating_decimal = 9 / 1100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l715_71512


namespace NUMINAMATH_CALUDE_skittles_eaten_l715_71540

/-- Proves that the number of Skittles eaten is the difference between initial and final amounts --/
theorem skittles_eaten (initial_skittles final_skittles : ℝ) (oranges_bought : ℝ) :
  initial_skittles = 7 →
  final_skittles = 2 →
  oranges_bought = 18 →
  initial_skittles - final_skittles = 5 := by
  sorry

end NUMINAMATH_CALUDE_skittles_eaten_l715_71540


namespace NUMINAMATH_CALUDE_insurance_claims_percentage_l715_71557

theorem insurance_claims_percentage (jan_claims missy_claims : ℕ) 
  (h1 : jan_claims = 20)
  (h2 : missy_claims = 41)
  (h3 : missy_claims = jan_claims + 15 + (jan_claims * 30 / 100)) :
  ∃ (john_claims : ℕ), 
    missy_claims = john_claims + 15 ∧ 
    john_claims = jan_claims + (jan_claims * 30 / 100) := by
  sorry

end NUMINAMATH_CALUDE_insurance_claims_percentage_l715_71557


namespace NUMINAMATH_CALUDE_adrianna_gum_purchase_l715_71581

/-- The number of gum pieces Adrianna bought in the second store visit -/
def second_store_purchase (initial_gum : ℕ) (first_store_purchase : ℕ) (total_friends : ℕ) : ℕ :=
  total_friends - (initial_gum + first_store_purchase)

/-- Theorem: Given Adrianna's initial 10 pieces of gum, 3 pieces bought from the first store,
    and 15 friends who received gum, the number of gum pieces bought in the second store visit is 2. -/
theorem adrianna_gum_purchase :
  second_store_purchase 10 3 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_purchase_l715_71581


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l715_71534

theorem complex_fraction_sum : 
  (Complex.I : ℂ) ^ 2 = -1 → 
  (7 + 3 * Complex.I) / (7 - 3 * Complex.I) + (7 - 3 * Complex.I) / (7 + 3 * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l715_71534


namespace NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l715_71508

theorem square_sum_plus_sum_squares : (6 + 10)^2 + (6^2 + 10^2) = 392 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l715_71508


namespace NUMINAMATH_CALUDE_odd_function_implies_m_zero_l715_71501

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + 3mx^2 + nx + m^2 -/
def f (m n : ℝ) (x : ℝ) : ℝ :=
  x^3 + 3*m*x^2 + n*x + m^2

theorem odd_function_implies_m_zero (m n : ℝ) :
  IsOdd (f m n) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_zero_l715_71501


namespace NUMINAMATH_CALUDE_max_x_minus_y_on_circle_l715_71597

theorem max_x_minus_y_on_circle :
  ∀ x y : ℝ, x^2 + y^2 - 4*x - 2*y - 4 = 0 →
  ∃ z : ℝ, z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ w : ℝ, (∃ a b : ℝ, a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_on_circle_l715_71597


namespace NUMINAMATH_CALUDE_slope_of_BF_l715_71598

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 
  ∃ m : ℝ, y + 2 = m * (x + 3)

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem slope_of_BF (B : ℝ × ℝ) :
  parabola B.1 B.2 →
  tangent_line B.1 B.2 →
  second_quadrant B.1 B.2 →
  (B.2 - focus.2) / (B.1 - focus.1) = -3/4 :=
sorry

end NUMINAMATH_CALUDE_slope_of_BF_l715_71598


namespace NUMINAMATH_CALUDE_product_sum_relation_l715_71596

theorem product_sum_relation (a b c N : ℕ) : 
  0 < a → 0 < b → 0 < c →
  a < b → b < c →
  c = a + b →
  N = a * b * c →
  N = 8 * (a + b + c) →
  N = 160 := by
sorry

end NUMINAMATH_CALUDE_product_sum_relation_l715_71596


namespace NUMINAMATH_CALUDE_point_on_linear_function_l715_71542

theorem point_on_linear_function (m : ℝ) : 
  (3 : ℝ) = 2 * m + 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_function_l715_71542


namespace NUMINAMATH_CALUDE_f_composition_negative_one_l715_71572

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (x + 3) else Real.log x

theorem f_composition_negative_one : f (f (-1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_one_l715_71572


namespace NUMINAMATH_CALUDE_a_range_l715_71566

/-- Set A defined as { x | a ≤ x ≤ a+3 } -/
def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 3 }

/-- Set B defined as { x | x < -1 or x > 5 } -/
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

/-- Theorem stating that if A ∪ B = B, then a is in (-∞, -4) ∪ (5, +∞) -/
theorem a_range (a : ℝ) : (A a ∪ B = B) → a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l715_71566


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l715_71563

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x)^(1/3 : ℝ) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l715_71563


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l715_71537

theorem abs_m_minus_n_equals_five (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l715_71537


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l715_71589

/-- Represents a rectangular yard with flower beds and a trapezoidal lawn -/
structure YardWithFlowerBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  trapezoid_height : ℝ
  num_flower_beds : ℕ

/-- The fraction of the yard occupied by flower beds -/
def flower_bed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  25 / 324

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction_is_correct (yard : YardWithFlowerBeds) 
    (h1 : yard.trapezoid_short_side = 26)
    (h2 : yard.trapezoid_long_side = 36)
    (h3 : yard.trapezoid_height = 6)
    (h4 : yard.num_flower_beds = 3) : 
  flower_bed_fraction yard = 25 / 324 := by
  sorry

#check flower_bed_fraction_is_correct

end NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l715_71589


namespace NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l715_71514

theorem square_triangle_equal_perimeter (x : ℝ) : 
  (4 * (x + 2) = 3 * (2 * x)) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l715_71514


namespace NUMINAMATH_CALUDE_exponent_division_l715_71574

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end NUMINAMATH_CALUDE_exponent_division_l715_71574


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l715_71527

theorem percentage_error_calculation : 
  let incorrect_factor : ℚ := 3 / 5
  let correct_factor : ℚ := 5 / 3
  let ratio := incorrect_factor / correct_factor
  let error_percentage := (1 - ratio) * 100
  error_percentage = 64 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l715_71527
