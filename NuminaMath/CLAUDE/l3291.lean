import Mathlib

namespace pie_division_l3291_329102

theorem pie_division (total_pie : ℚ) (people : ℕ) :
  total_pie = 5 / 8 ∧ people = 4 →
  total_pie / people = 5 / 32 := by
sorry

end pie_division_l3291_329102


namespace coordinates_of_point_E_l3291_329178

/-- Given points A, B, C, D, and E in the plane, where D lies on line AB and E is on the extension of DC,
    prove that E has specific coordinates. -/
theorem coordinates_of_point_E (A B C D E : ℝ × ℝ) : 
  A = (-2, 1) →
  B = (1, 4) →
  C = (4, -3) →
  (∃ t : ℝ, D = (1 - t) • A + t • B ∧ t = 2/3) →
  (∃ s : ℝ, E = (1 + s) • D - s • C ∧ s = 5) →
  E = (-8/3, 11/3) := by
sorry

end coordinates_of_point_E_l3291_329178


namespace middle_number_proof_l3291_329113

theorem middle_number_proof (a b c : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_order : a < b ∧ b < c)
  (h_sum_ab : a + b = 18)
  (h_sum_ac : a + c = 22)
  (h_sum_bc : b + c = 26)
  (h_diff : c - a = 10) : 
  b = 11 := by sorry

end middle_number_proof_l3291_329113


namespace judy_hits_percentage_l3291_329147

theorem judy_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h_total : total_hits = 35)
  (h_home : home_runs = 1)
  (h_triple : triples = 1)
  (h_double : doubles = 5) :
  (total_hits - (home_runs + triples + doubles)) / total_hits = 4/5 := by
sorry

end judy_hits_percentage_l3291_329147


namespace yellow_hard_hats_l3291_329104

def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def carl_takes_pink : ℕ := 4
def john_takes_pink : ℕ := 6
def john_takes_green : ℕ := 2 * john_takes_pink
def remaining_total : ℕ := 43

theorem yellow_hard_hats (initial_yellow : ℕ) :
  initial_pink - carl_takes_pink - john_takes_pink +
  initial_green - john_takes_green +
  initial_yellow = remaining_total →
  initial_yellow = 24 :=
by sorry

end yellow_hard_hats_l3291_329104


namespace product_evaluation_l3291_329175

theorem product_evaluation (n : ℤ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end product_evaluation_l3291_329175


namespace james_total_earnings_l3291_329116

def january_earnings : ℕ := 4000

def february_earnings : ℕ := 2 * january_earnings

def march_earnings : ℕ := february_earnings - 2000

def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings : total_earnings = 18000 := by
  sorry

end james_total_earnings_l3291_329116


namespace percentage_difference_l3291_329158

theorem percentage_difference (x y : ℝ) (h1 : y = 125 * (1 + 0.1)) (h2 : x = 123.75) :
  (y - x) / y * 100 = 10 := by
  sorry

end percentage_difference_l3291_329158


namespace maria_paper_count_l3291_329154

/-- Represents the number of sheets of paper -/
structure PaperCount where
  whole : ℕ
  half : ℕ

/-- Calculates the remaining papers after giving away and folding -/
def remaining_papers (desk : ℕ) (backpack : ℕ) (given_away : ℕ) (folded : ℕ) : PaperCount :=
  { whole := desk + backpack - given_away - folded,
    half := folded }

theorem maria_paper_count : 
  ∀ (x y : ℕ), x ≤ 91 → y ≤ 91 - x → 
  remaining_papers 50 41 x y = { whole := 91 - x - y, half := y } := by
sorry

end maria_paper_count_l3291_329154


namespace ellipse_eccentricity_l3291_329130

theorem ellipse_eccentricity (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (k + 8) + y^2 / 9 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 - b^2) / a^2 = 1/4) →  -- Eccentricity condition
  (k = 4 ∨ k = -5/4) :=
by sorry

end ellipse_eccentricity_l3291_329130


namespace equation_solutions_l3291_329119

noncomputable def solution_equation (a b c d x : ℝ) : Prop :=
  (a*x + b) / (a + b*x) + (c*x + d) / (c + d*x) = 
  (a*x - b) / (a - b*x) + (c*x - d) / (c - d*x)

theorem equation_solutions 
  (a b c d : ℝ) 
  (h1 : a*d + b*c ≠ 0) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : d ≠ 0) :
  (∀ x : ℝ, x ≠ a/b ∧ x ≠ -a/b ∧ x ≠ c/d ∧ x ≠ -c/d →
    (x = 1 ∨ x = -1 ∨ x = Real.sqrt (a*c/(b*d)) ∨ x = -Real.sqrt (a*c/(b*d))) ↔ 
    solution_equation a b c d x) :=
sorry

end equation_solutions_l3291_329119


namespace difference_of_same_prime_divisors_l3291_329133

/-- For any natural number, there exist two natural numbers with the same number of distinct prime divisors whose difference is the original number. -/
theorem difference_of_same_prime_divisors (n : ℕ) : 
  ∃ a b : ℕ, n = a - b ∧ (Finset.card (Nat.factorization a).support = Finset.card (Nat.factorization b).support) := by
  sorry

end difference_of_same_prime_divisors_l3291_329133


namespace f_equals_g_l3291_329179

-- Define the two functions
def f (x : ℝ) : ℝ := (x ^ (1/3)) ^ 3
def g (x : ℝ) : ℝ := x

-- Theorem statement
theorem f_equals_g : ∀ (x : ℝ), f x = g x := by
  sorry

end f_equals_g_l3291_329179


namespace arithmetic_sequence_property_l3291_329126

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  is_positive : ∀ n, a n > 0

/-- Theorem: In an arithmetic sequence with positive terms, if 2a₆ + 2a₈ = a₇², then a₇ = 4 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : 2 * seq.a 6 + 2 * seq.a 8 = (seq.a 7) ^ 2) : 
    seq.a 7 = 4 := by
  sorry

end arithmetic_sequence_property_l3291_329126


namespace inequality_equivalence_l3291_329197

theorem inequality_equivalence (x : ℝ) : 
  3/20 + |2*x - 5/40| < 9/40 ↔ 1/40 < x ∧ x < 1/10 := by
sorry

end inequality_equivalence_l3291_329197


namespace multiplication_mistake_l3291_329100

theorem multiplication_mistake (x : ℝ) : 973 * x - 739 * x = 110305 → x = 471.4 := by
  sorry

end multiplication_mistake_l3291_329100


namespace knowledge_competition_probabilities_l3291_329151

/-- Represents the types of questions available in the competition -/
inductive QuestionType
  | Easy1
  | Easy2
  | Medium
  | Hard

/-- The point value associated with each question type -/
def pointValue : QuestionType → ℕ
  | QuestionType.Easy1 => 10
  | QuestionType.Easy2 => 10
  | QuestionType.Medium => 20
  | QuestionType.Hard => 40

/-- The probability of selecting each question type -/
def selectionProbability : QuestionType → ℚ
  | _ => 1/4

theorem knowledge_competition_probabilities :
  let differentValueProb := 1 - (2/4 * 2/4 + 1/4 * 1/4 + 1/4 * 1/4)
  let greaterValueProb := 1/4 * 2/4 + 1/4 * 3/4
  differentValueProb = 5/8 ∧ greaterValueProb = 5/16 := by
  sorry

#check knowledge_competition_probabilities

end knowledge_competition_probabilities_l3291_329151


namespace derivative_f_at_one_l3291_329145

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * Real.log x

theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 + 2 := by sorry

end derivative_f_at_one_l3291_329145


namespace zeros_product_less_than_one_l3291_329180

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem zeros_product_less_than_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂ → x₁ * x₂ < 1 := by
  sorry

end zeros_product_less_than_one_l3291_329180


namespace base4_arithmetic_theorem_l3291_329125

/-- Converts a number from base 4 to base 10 --/
def base4To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10To4 (n : ℕ) : ℕ := sorry

/-- Performs arithmetic operations in base 4 --/
def base4Arithmetic (a b c d : ℕ) : ℕ :=
  let a10 := base4To10 a
  let b10 := base4To10 b
  let c10 := base4To10 c
  let d10 := base4To10 d
  let result := a10 * b10 / c10 * d10
  base10To4 result

theorem base4_arithmetic_theorem :
  base4Arithmetic 231 21 3 2 = 10232 := by sorry

end base4_arithmetic_theorem_l3291_329125


namespace earl_stuffing_rate_l3291_329131

-- Define Earl's stuffing rate (envelopes per minute)
def earl_rate : ℝ := sorry

-- Define Ellen's stuffing rate (envelopes per minute)
def ellen_rate : ℝ := sorry

-- Condition 1: Ellen's rate is 2/3 of Earl's rate
axiom rate_relation : ellen_rate = (2/3) * earl_rate

-- Condition 2: Together they stuff 360 envelopes in 6 minutes
axiom combined_rate : earl_rate + ellen_rate = 360 / 6

-- Theorem to prove
theorem earl_stuffing_rate : earl_rate = 36 := by sorry

end earl_stuffing_rate_l3291_329131


namespace f_neg_two_f_is_even_l3291_329152

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2 * abs x + 2

-- Define the domain
def domain : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 5}

-- Theorem 1: f(-2) = 10
theorem f_neg_two : f (-2) = 10 := by sorry

-- Theorem 2: f is an even function on the domain
theorem f_is_even : ∀ x ∈ domain, f (-x) = f x := by sorry

end f_neg_two_f_is_even_l3291_329152


namespace intersection_range_l3291_329128

theorem intersection_range (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y = 2 ∧ x^2 / 6 + y^2 / 2 = 1) →
  a / b ∈ Set.Iic (-1) ∪ Set.Ici 1 := by
sorry

end intersection_range_l3291_329128


namespace walking_time_calculation_l3291_329169

/-- Given a person walking at a constant rate who covers 45 meters in 15 minutes,
    prove that it will take 30 minutes to cover an additional 90 meters. -/
theorem walking_time_calculation (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
    (h1 : initial_distance = 45)
    (h2 : initial_time = 15)
    (h3 : additional_distance = 90) :
    additional_distance / (initial_distance / initial_time) = 30 := by
  sorry


end walking_time_calculation_l3291_329169


namespace flour_difference_l3291_329138

theorem flour_difference : (7 : ℚ) / 8 - (5 : ℚ) / 6 = (1 : ℚ) / 24 := by
  sorry

end flour_difference_l3291_329138


namespace log_function_fixed_point_l3291_329107

-- Define the set of valid 'a' values
def ValidA := {a : ℝ | a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1}

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a + 2

-- Theorem statement
theorem log_function_fixed_point (a : ℝ) (ha : a ∈ ValidA) :
  f a 1 = 2 :=
sorry

end log_function_fixed_point_l3291_329107


namespace perpendicular_line_equation_l3291_329132

/-- Given a line l: x + 2y + p = 0 (p ∈ ℝ), prove that 2x - y - 1 = 0 is the equation of the line
    passing through the point P(2,3) and perpendicular to l. -/
theorem perpendicular_line_equation (p : ℝ) :
  let l : ℝ → ℝ → Prop := fun x y ↦ x + 2 * y + p = 0
  let perpendicular_line : ℝ → ℝ → Prop := fun x y ↦ 2 * x - y - 1 = 0
  (∀ x y, l x y → (perpendicular_line x y → False) → False) ∧
  perpendicular_line 2 3 :=
by sorry

end perpendicular_line_equation_l3291_329132


namespace complex_equation_solution_l3291_329195

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (3 - 2 * i * z = 1 + 4 * i * z) ∧ (z = -i / 3) :=
by sorry

end complex_equation_solution_l3291_329195


namespace rectangular_prism_diagonal_l3291_329186

theorem rectangular_prism_diagonal (length width height : ℝ) 
  (h_length : length = 12) 
  (h_width : width = 15) 
  (h_height : height = 8) : 
  Real.sqrt (length^2 + width^2 + height^2) = Real.sqrt 433 :=
by sorry

end rectangular_prism_diagonal_l3291_329186


namespace highest_score_l3291_329164

theorem highest_score (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (bc_gt_ad : b + c > a + d)
  (a_gt_bd : a > b + d) :
  c > a ∧ c > b ∧ c > d :=
sorry

end highest_score_l3291_329164


namespace regression_estimate_l3291_329141

theorem regression_estimate :
  let regression_equation (x : ℝ) := 4.75 * x + 2.57
  regression_equation 28 = 135.57 := by sorry

end regression_estimate_l3291_329141


namespace sample_size_proof_l3291_329181

theorem sample_size_proof (n : ℕ) : 
  (∃ (x : ℚ), 
    x > 0 ∧ 
    2*x + 3*x + 4*x + 6*x + 4*x + x = 1 ∧ 
    (2*x + 3*x + 4*x) * n = 27) → 
  n = 60 := by
sorry

end sample_size_proof_l3291_329181


namespace probability_to_reach_target_is_correct_l3291_329123

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a possible move direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of each direction -/
def directionProbability : ℚ := 1 / 4

/-- The number of steps allowed -/
def numberOfSteps : ℕ := 6

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The target point -/
def targetPoint : Point := ⟨3, 1⟩

/-- Function to calculate the probability of reaching the target point -/
def probabilityToReachTarget (start : Point) (target : Point) (steps : ℕ) : ℚ :=
  sorry

theorem probability_to_reach_target_is_correct :
  probabilityToReachTarget startPoint targetPoint numberOfSteps = 45 / 1024 := by
  sorry

end probability_to_reach_target_is_correct_l3291_329123


namespace triangle_side_sum_l3291_329101

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) (side_c : ℝ) (h5 : side_c = 8) : 
  ∃ (side_a side_b : ℝ), 
    abs ((side_a + side_b) - 18.9) < 0.05 :=
by sorry

end triangle_side_sum_l3291_329101


namespace fifteen_equidistant_planes_spheres_l3291_329143

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of 5 points in 3D space -/
def FivePoints := Fin 5 → Point3D

/-- Predicate to check if 5 points lie on the same plane -/
def lieOnSamePlane (points : FivePoints) : Prop := sorry

/-- Predicate to check if 5 points lie on the same sphere -/
def lieOnSameSphere (points : FivePoints) : Prop := sorry

/-- Count of equidistant planes or spheres from 5 points -/
def countEquidistantPlanesSpheres (points : FivePoints) : ℕ := sorry

/-- Theorem stating that there are exactly 15 equidistant planes or spheres -/
theorem fifteen_equidistant_planes_spheres (points : FivePoints) 
  (h1 : ¬ lieOnSamePlane points) (h2 : ¬ lieOnSameSphere points) :
  countEquidistantPlanesSpheres points = 15 := by sorry

end fifteen_equidistant_planes_spheres_l3291_329143


namespace remainder_theorem_l3291_329136

theorem remainder_theorem : (9 * 7^18 + 2^18) % 9 = 1 := by
  sorry

end remainder_theorem_l3291_329136


namespace second_month_sale_l3291_329161

def average_sale : ℕ := 6800
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month3 : ℕ := 7230
def sale_month4 : ℕ := 6562
def sale_month5 : ℕ := 6791
def sale_month6 : ℕ := 6791

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = 13991 ∧
    (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / num_months = average_sale :=
by sorry

end second_month_sale_l3291_329161


namespace complement_A_intersect_B_l3291_329189

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {1, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {4} :=
sorry

end complement_A_intersect_B_l3291_329189


namespace percentage_of_120_to_80_l3291_329166

theorem percentage_of_120_to_80 : ∃ (p : ℝ), (120 : ℝ) / 80 * 100 = p ∧ p = 150 := by
  sorry

end percentage_of_120_to_80_l3291_329166


namespace inequality_and_equality_condition_l3291_329137

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  1 + a^2017 + b^2017 ≥ a^10 * b^7 + a^7 * b^2000 + a^2000 * b^10 ∧
  (1 + a^2017 + b^2017 = a^10 * b^7 + a^7 * b^2000 + a^2000 * b^10 ↔ a = 1 ∧ b = 1) :=
by sorry

end inequality_and_equality_condition_l3291_329137


namespace star_problem_l3291_329149

-- Define the star operation
def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

-- State the theorem
theorem star_problem : star (star 2 3) 2 = 3 + 2^31 := by
  sorry

end star_problem_l3291_329149


namespace quadratic_roots_range_l3291_329127

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + m*x + 1 = 0

-- Define the property of having two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  m < -2 ∨ m > 2

-- State the theorem
theorem quadratic_roots_range :
  ∀ m : ℝ, has_two_distinct_real_roots m ↔ m_range m :=
by sorry

end quadratic_roots_range_l3291_329127


namespace consecutive_sum_equals_50_l3291_329103

/-- The sum of consecutive integers from a given start to an end -/
def sum_consecutive (start : Int) (count : Nat) : Int :=
  count * (2 * start + count.pred) / 2

/-- Proves that there are exactly 100 consecutive integers starting from -49 whose sum is 50 -/
theorem consecutive_sum_equals_50 : ∃! n : Nat, sum_consecutive (-49) n = 50 ∧ n > 0 := by
  sorry

end consecutive_sum_equals_50_l3291_329103


namespace amelias_dinner_l3291_329173

/-- Amelia's dinner problem -/
theorem amelias_dinner (first_course second_course dessert remaining_money : ℝ) 
  (h1 : first_course = 15)
  (h2 : second_course = first_course + 5)
  (h3 : dessert = 0.25 * second_course)
  (h4 : remaining_money = 20) : 
  first_course + second_course + dessert + remaining_money = 60 := by
  sorry

end amelias_dinner_l3291_329173


namespace stuffed_animals_theorem_l3291_329114

/-- Represents the number of stuffed animals for each girl -/
structure StuffedAnimals where
  mckenna : ℕ
  kenley : ℕ
  tenly : ℕ

/-- Calculates the total number of stuffed animals -/
def total (sa : StuffedAnimals) : ℕ :=
  sa.mckenna + sa.kenley + sa.tenly

/-- Calculates the average number of stuffed animals per girl -/
def average (sa : StuffedAnimals) : ℚ :=
  (total sa : ℚ) / 3

/-- Calculates the percentage of total stuffed animals McKenna has -/
def mckennaPercentage (sa : StuffedAnimals) : ℚ :=
  (sa.mckenna : ℚ) / (total sa : ℚ) * 100

theorem stuffed_animals_theorem (sa : StuffedAnimals) 
  (h1 : sa.mckenna = 34)
  (h2 : sa.kenley = 2 * sa.mckenna)
  (h3 : sa.tenly = sa.kenley + 5) :
  total sa = 175 ∧ 
  58.32 < average sa ∧ average sa < 58.34 ∧
  19.42 < mckennaPercentage sa ∧ mckennaPercentage sa < 19.44 := by
  sorry

#eval total { mckenna := 34, kenley := 68, tenly := 73 }
#eval average { mckenna := 34, kenley := 68, tenly := 73 }
#eval mckennaPercentage { mckenna := 34, kenley := 68, tenly := 73 }

end stuffed_animals_theorem_l3291_329114


namespace bake_sale_total_l3291_329157

/-- Represents the number of cookies sold at a bake sale -/
structure CookieSale where
  raisin : ℕ
  oatmeal : ℕ
  chocolate_chip : ℕ

/-- Theorem stating the total number of cookies sold given the conditions -/
theorem bake_sale_total (sale : CookieSale) : 
  sale.raisin = 42 ∧ 
  sale.raisin = 6 * sale.oatmeal ∧ 
  sale.raisin = 2 * sale.chocolate_chip → 
  sale.raisin + sale.oatmeal + sale.chocolate_chip = 70 := by
  sorry

#check bake_sale_total

end bake_sale_total_l3291_329157


namespace sum_of_series_equals_half_l3291_329122

theorem sum_of_series_equals_half :
  let series := λ k : ℕ => (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1)
  ∑' k, series k = 1 / 2 := by
  sorry

end sum_of_series_equals_half_l3291_329122


namespace iron_conducts_electricity_l3291_329110

-- Define the universe of discourse
variable (Object : Type)

-- Define predicates
variable (is_metal : Object → Prop)
variable (conducts_electricity : Object → Prop)

-- Define iron as a constant
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity 
  (all_metals_conduct : ∀ x, is_metal x → conducts_electricity x) 
  (iron_is_metal : is_metal iron) : 
  conducts_electricity iron := by
  sorry

end iron_conducts_electricity_l3291_329110


namespace smallest_n_with_6474_l3291_329106

def contains_subsequence (s t : List Nat) : Prop :=
  ∃ i, t = s.drop i ++ s.take i

def digits_to_list (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

def concatenate_digits (a b c : Nat) : List Nat :=
  (digits_to_list a) ++ (digits_to_list b) ++ (digits_to_list c)

theorem smallest_n_with_6474 :
  ∀ n : Nat, n < 46 →
    ¬(contains_subsequence (concatenate_digits n (n+1) (n+2)) [6,4,7,4]) ∧
  contains_subsequence (concatenate_digits 46 47 48) [6,4,7,4] :=
sorry

end smallest_n_with_6474_l3291_329106


namespace characterization_of_m_l3291_329140

-- Define a good integer
def is_good (n : ℤ) : Prop :=
  ¬ ∃ k : ℤ, n.natAbs = k^2

-- Define the property for m
def has_property (m : ℤ) : Prop :=
  ∀ N : ℕ, ∃ a b c : ℤ,
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (is_good a ∧ is_good b ∧ is_good c) ∧
    (a + b + c = m) ∧
    (∃ k : ℤ, a * b * c = (2*k + 1)^2) ∧
    (N < a.natAbs ∧ N < b.natAbs ∧ N < c.natAbs)

-- The main theorem
theorem characterization_of_m (m : ℤ) :
  has_property m ↔ m % 4 = 3 :=
sorry

end characterization_of_m_l3291_329140


namespace percentage_problem_l3291_329105

theorem percentage_problem (P : ℝ) : 
  (1/10 * 7000 - P/100 * 7000 = 700) → P = 0 :=
by sorry

end percentage_problem_l3291_329105


namespace parallelogram_EFGH_area_l3291_329199

/-- Represents a parallelogram with a base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.base * p.height

/-- Theorem: The area of parallelogram EFGH is 18 square units -/
theorem parallelogram_EFGH_area :
  let p : Parallelogram := { base := 6, height := 3 }
  area p = 18 := by
  sorry

end parallelogram_EFGH_area_l3291_329199


namespace urn_probability_theorem_l3291_329170

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Performs one operation on the urn state -/
def performOperation (state : UrnState) : UrnState :=
  sorry

/-- Calculates the probability of drawing a specific color -/
def drawProbability (state : UrnState) (color : Color) : ℚ :=
  sorry

/-- Calculates the probability of a specific sequence of draws -/
def sequenceProbability (sequence : List Color) : ℚ :=
  sorry

/-- Counts the number of valid sequences resulting in 3 red and 3 blue balls -/
def countValidSequences : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initialState : UrnState := ⟨1, 1⟩
  let finalState : UrnState := ⟨3, 3⟩
  let numOperations : ℕ := 5
  (countValidSequences * sequenceProbability [Color.Red, Color.Red, Color.Red, Color.Blue, Color.Blue]) = 1 / 6 :=
sorry

end urn_probability_theorem_l3291_329170


namespace area_of_larger_rectangle_l3291_329171

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The area of a larger rectangle formed by six identical smaller rectangles -/
def largerRectangleArea (smallRect : Rectangle) : ℝ :=
  (3 * smallRect.width) * (2 * smallRect.length)

theorem area_of_larger_rectangle :
  ∀ (smallRect : Rectangle),
    smallRect.length = 2 * smallRect.width →
    smallRect.length + smallRect.width = 21 →
    largerRectangleArea smallRect = 588 := by
  sorry

end area_of_larger_rectangle_l3291_329171


namespace triangle_theorem_l3291_329162

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the main theorem
theorem triangle_theorem (ABC : Triangle) 
  (h1 : (Real.cos ABC.B - 2 * Real.cos ABC.A) / (2 * ABC.a - ABC.b) = Real.cos ABC.C / ABC.c) :
  -- Part 1: a/b = 2
  ABC.a / ABC.b = 2 ∧
  -- Part 2: If angle A is obtuse and c = 3, then 0 < b < 3
  (ABC.A > Real.pi / 2 ∧ ABC.c = 3 → 0 < ABC.b ∧ ABC.b < 3) :=
by sorry


end triangle_theorem_l3291_329162


namespace special_sequence_eventually_periodic_l3291_329108

/-- A sequence of positive integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (a 1 < a 2) ∧
  (∀ n ≥ 3,
    (a n > a (n-1)) ∧
    (∃! (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n-1 ∧ a n = a i + a j) ∧
    (∀ m < n, (∃ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ m-1 ∧ a m = a i + a j) → a n > a m))

/-- The set of even numbers in the sequence is finite -/
def FinitelyManyEven (a : ℕ → ℕ) : Prop :=
  ∃ (S : Finset ℕ), ∀ n, Even (a n) → n ∈ S

/-- The sequence of differences is eventually periodic -/
def EventuallyPeriodic (s : ℕ → ℕ) : Prop :=
  ∃ (k p : ℕ), p > 0 ∧ ∀ n ≥ k, s (n + p) = s n

/-- The main theorem -/
theorem special_sequence_eventually_periodic (a : ℕ → ℕ)
  (h1 : SpecialSequence a) (h2 : FinitelyManyEven a) :
  EventuallyPeriodic (fun n => a (n+1) - a n) :=
sorry

end special_sequence_eventually_periodic_l3291_329108


namespace max_terms_sum_to_target_l3291_329163

/-- The sequence of odd numbers from 1 to 101 -/
def oddSequence : List Nat := List.range 51 |>.map (fun n => 2*n + 1)

/-- The sum we're aiming for -/
def targetSum : Nat := 1949

/-- The maximum number of terms that sum to the target -/
def maxTerms : Nat := 44

theorem max_terms_sum_to_target :
  ∃ (subset : List Nat),
    subset.toFinset ⊆ oddSequence.toFinset ∧
    subset.sum = targetSum ∧
    subset.length = maxTerms ∧
    ∀ (otherSubset : List Nat),
      otherSubset.toFinset ⊆ oddSequence.toFinset →
      otherSubset.sum = targetSum →
      otherSubset.length ≤ maxTerms :=
by sorry

end max_terms_sum_to_target_l3291_329163


namespace expand_polynomial_l3291_329121

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_polynomial_l3291_329121


namespace tangent_circle_equation_l3291_329156

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The point P -/
def point_P : ℝ × ℝ := (-1, 0)

/-- The center of circle C -/
def center_C : ℝ × ℝ := (1, 2)

/-- The equation of the circle passing through the tangency points and the center of C -/
def target_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

/-- The theorem stating that the target circle passes through the tangency points and the center of C -/
theorem tangent_circle_equation : 
  ∃ (A B : ℝ × ℝ), 
    (∀ x y, circle_C x y → ((x - point_P.1)^2 + (y - point_P.2)^2 ≤ (A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)) ∧
    (∀ x y, circle_C x y → ((x - point_P.1)^2 + (y - point_P.2)^2 ≤ (B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) ∧
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    target_circle A.1 A.2 ∧
    target_circle B.1 B.2 ∧
    target_circle center_C.1 center_C.2 :=
sorry

end tangent_circle_equation_l3291_329156


namespace gcd_35_and_number_between_80_90_l3291_329160

theorem gcd_35_and_number_between_80_90 :
  ∃! n : ℕ, 80 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 35 n = 7 :=
by sorry

end gcd_35_and_number_between_80_90_l3291_329160


namespace brownie_cost_l3291_329120

/-- The cost of each brownie at Tamara's bake sale -/
theorem brownie_cost (total_revenue : ℚ) (num_pans : ℕ) (pieces_per_pan : ℕ) 
  (h1 : total_revenue = 32)
  (h2 : num_pans = 2)
  (h3 : pieces_per_pan = 8) :
  total_revenue / (num_pans * pieces_per_pan) = 2 := by
  sorry

end brownie_cost_l3291_329120


namespace special_triangle_perimeter_l3291_329111

/-- A triangle with integer side lengths satisfying specific conditions -/
structure SpecialTriangle where
  x : ℕ
  y : ℕ
  side_product : x * y = 105
  triangle_inequality : x + y > 13 ∧ x + 13 > y ∧ y + 13 > x

/-- The perimeter of the special triangle is 35 -/
theorem special_triangle_perimeter (t : SpecialTriangle) : 13 + t.x + t.y = 35 := by
  sorry

#check special_triangle_perimeter

end special_triangle_perimeter_l3291_329111


namespace system_of_equations_solution_l3291_329192

theorem system_of_equations_solution (x y m : ℝ) : 
  (3 * x - y = 4 * m + 1) → 
  (x + y = 2 * m - 5) → 
  (x - y = 4) → 
  (m = 1) := by
sorry

end system_of_equations_solution_l3291_329192


namespace system_solutions_l3291_329139

/-- The system of equations -/
def satisfies_system (a b c : ℝ) : Prop :=
  a^5 = 5*b^3 - 4*c ∧ b^5 = 5*c^3 - 4*a ∧ c^5 = 5*a^3 - 4*b

/-- The set of solutions -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 1, 1), (-1, -1, -1), (2, 2, 2), (-2, -2, -2)}

/-- The main theorem -/
theorem system_solutions :
  ∀ (a b c : ℝ), satisfies_system a b c ↔ (a, b, c) ∈ solution_set := by
  sorry

end system_solutions_l3291_329139


namespace loss_percentage_proof_l3291_329194

def calculate_loss_percentage (cost_prices selling_prices : List ℚ) : ℚ :=
  let total_cp := cost_prices.sum
  let total_sp := selling_prices.sum
  let loss := total_cp - total_sp
  (loss / total_cp) * 100

theorem loss_percentage_proof (cost_prices selling_prices : List ℚ) :
  cost_prices = [1200, 1500, 1800] →
  selling_prices = [800, 1300, 1500] →
  calculate_loss_percentage cost_prices selling_prices = 20 := by
  sorry

#eval calculate_loss_percentage [1200, 1500, 1800] [800, 1300, 1500]

end loss_percentage_proof_l3291_329194


namespace jeremy_payment_l3291_329155

/-- The total amount owed to Jeremy for cleaning rooms and washing windows -/
theorem jeremy_payment (room_rate : ℚ) (window_rate : ℚ) (rooms_cleaned : ℚ) (windows_washed : ℚ)
  (h1 : room_rate = 13 / 3)
  (h2 : window_rate = 5 / 2)
  (h3 : rooms_cleaned = 8 / 5)
  (h4 : windows_washed = 11 / 4) :
  room_rate * rooms_cleaned + window_rate * windows_washed = 553 / 40 :=
by sorry

end jeremy_payment_l3291_329155


namespace student_arrangement_l3291_329142

/-- Number of ways to arrange n distinct objects from m objects -/
def arrangement (m n : ℕ) : ℕ := sorry

/-- The number of students -/
def total_students : ℕ := 14

/-- The number of female students -/
def female_students : ℕ := 6

/-- The number of male students -/
def male_students : ℕ := 8

/-- The number of female students that must be grouped together -/
def grouped_females : ℕ := 4

/-- The number of gaps after arranging male students and grouped females -/
def gaps : ℕ := male_students + 1

theorem student_arrangement :
  arrangement male_students male_students *
  arrangement gaps (female_students - grouped_females) *
  arrangement grouped_females grouped_females =
  arrangement total_students total_students := by sorry

end student_arrangement_l3291_329142


namespace polynomial_factorization_l3291_329118

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8) := by
  sorry

end polynomial_factorization_l3291_329118


namespace cos_A_value_c_value_l3291_329187

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 2 * t.A ∧ Real.cos t.C = 1/8

-- Theorem 1: Prove cos A = 3/4
theorem cos_A_value (t : Triangle) (h : triangle_conditions t) : Real.cos t.A = 3/4 := by
  sorry

-- Theorem 2: Prove c = 6 when a = 4
theorem c_value (t : Triangle) (h1 : triangle_conditions t) (h2 : t.a = 4) : t.c = 6 := by
  sorry

end cos_A_value_c_value_l3291_329187


namespace tv_watching_time_l3291_329188

/-- Given children watch 6 hours of television in 2 weeks and are allowed to watch 4 days a week,
    prove they spend 45 minutes each day watching television. -/
theorem tv_watching_time (hours_per_two_weeks : ℕ) (days_per_week : ℕ) 
    (h1 : hours_per_two_weeks = 6) 
    (h2 : days_per_week = 4) : 
  (hours_per_two_weeks * 60) / (days_per_week * 2) = 45 := by
  sorry

end tv_watching_time_l3291_329188


namespace book_purchase_total_price_l3291_329129

theorem book_purchase_total_price
  (total_books : ℕ)
  (math_books : ℕ)
  (math_book_price : ℕ)
  (history_book_price : ℕ)
  (h1 : total_books = 80)
  (h2 : math_books = 27)
  (h3 : math_book_price = 4)
  (h4 : history_book_price = 5) :
  let history_books := total_books - math_books
  let total_price := math_books * math_book_price + history_books * history_book_price
  total_price = 373 := by
sorry

end book_purchase_total_price_l3291_329129


namespace range_of_g_l3291_329115

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.sin x ^ 2 + Real.cos x ^ 4 ∧ Real.sin x ^ 2 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end range_of_g_l3291_329115


namespace cn_relation_sqrt_c_equals_c8_l3291_329153

-- Define cn as a function that returns a natural number with n ones
def cn (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Define the relation between cn and cn+1
theorem cn_relation (n : ℕ) : cn (n + 1) = 10 * cn n + 1 := by sorry

-- Define c
def c : ℕ := 123456787654321

-- Theorem to prove
theorem sqrt_c_equals_c8 : ∃ (x : ℕ), x * x = c ∧ x = cn 8 := by sorry

end cn_relation_sqrt_c_equals_c8_l3291_329153


namespace sin_135_degrees_l3291_329159

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l3291_329159


namespace ellipse_and_line_properties_l3291_329167

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Defines a line passing through two points -/
structure Line where
  m : ℝ
  c : ℝ

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties
  (M : Ellipse)
  (h_eccentricity : M.a^2 - M.b^2 = M.a^2 / 2)
  (AB : Line)
  (h_AB_points : AB.m * (-M.a) + AB.c = 0 ∧ AB.c = M.b)
  (h_AB_distance : (M.a * M.b / Real.sqrt (M.a^2 + M.b^2))^2 = 2/3)
  (l : Line)
  (h_l_point : l.c = -1)
  (h_intersection_ratio : ∃ (y₁ y₂ : ℝ), y₁ = -3 * y₂ ∧
    y₁ + y₂ = -2 * l.m / (l.m^2 + 2) ∧
    y₁ * y₂ = -1 / (l.m^2 + 2)) :
  M.a^2 = 2 ∧ M.b^2 = 1 ∧ l.m = 1 := by sorry

end ellipse_and_line_properties_l3291_329167


namespace heptagon_foldable_to_quadrilateral_l3291_329150

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List Point2D

/-- A function to check if a polygon is convex -/
def isConvex (p : Polygon) : Prop := sorry

/-- A function to check if a polygon can be folded into a two-layered quadrilateral -/
def canFoldToTwoLayeredQuadrilateral (p : Polygon) : Prop := sorry

/-- Theorem: There exists a convex heptagon that can be folded into a two-layered quadrilateral -/
theorem heptagon_foldable_to_quadrilateral :
  ∃ (h : Polygon), h.vertices.length = 7 ∧ isConvex h ∧ canFoldToTwoLayeredQuadrilateral h := by
  sorry

end heptagon_foldable_to_quadrilateral_l3291_329150


namespace cube_coloring_l3291_329190

theorem cube_coloring (n : ℕ) (h : n > 0) : 
  (∃ (W B : ℕ), W + B = n^3 ∧ 
   3 * W = 3 * B ∧ 
   2 * W = n^3) → 
  ∃ k : ℕ, n = 2 * k :=
by sorry

end cube_coloring_l3291_329190


namespace inequality_proof_l3291_329172

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end inequality_proof_l3291_329172


namespace not_red_ball_percentage_is_52_5_percent_l3291_329174

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  cube_percentage : ℝ
  red_ball_percentage : ℝ

/-- Calculates the percentage of objects in the urn that are not red balls -/
def not_red_ball_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.cube_percentage) * (1 - urn.red_ball_percentage)

/-- Theorem stating that the percentage of objects in the urn that are not red balls is 52.5% -/
theorem not_red_ball_percentage_is_52_5_percent (urn : UrnComposition)
  (h1 : urn.cube_percentage = 0.3)
  (h2 : urn.red_ball_percentage = 0.25) :
  not_red_ball_percentage urn = 0.525 := by
  sorry

end not_red_ball_percentage_is_52_5_percent_l3291_329174


namespace vector_magnitude_l3291_329144

/-- Given two planar vectors a and b, prove that |2a - b| = 2√3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 3/5 ∧ a.2 = -4/5) →  -- Vector a = (3/5, -4/5)
  (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1) →  -- |a| = 1
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 2) →  -- |b| = 2
  (a.1 * b.1 + a.2 * b.2 = -1) →  -- a · b = -1 (dot product for 120° angle)
  Real.sqrt (((2 * a.1 - b.1) ^ 2) + ((2 * a.2 - b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end vector_magnitude_l3291_329144


namespace cylinder_lateral_area_l3291_329124

/-- The lateral area of a cylinder with volume π and base radius 1 is 2π -/
theorem cylinder_lateral_area (V : ℝ) (r : ℝ) (h : ℝ) : 
  V = π → r = 1 → V = π * r^2 * h → 2 * π * r * h = 2 * π := by
  sorry

end cylinder_lateral_area_l3291_329124


namespace max_bananas_purchase_l3291_329183

def apple_cost : ℕ := 3
def orange_cost : ℕ := 5
def banana_cost : ℕ := 8
def total_budget : ℕ := 100

def is_valid_purchase (apples oranges bananas : ℕ) : Prop :=
  apples ≥ 1 ∧ oranges ≥ 1 ∧ bananas ≥ 1 ∧
  apple_cost * apples + orange_cost * oranges + banana_cost * bananas ≤ total_budget

theorem max_bananas_purchase :
  ∃ (apples oranges : ℕ),
    is_valid_purchase apples oranges 11 ∧
    ∀ (a o b : ℕ), is_valid_purchase a o b → b ≤ 11 :=
by sorry

end max_bananas_purchase_l3291_329183


namespace candy_mix_proof_l3291_329168

/-- Proves that mixing 1 pound of candy A with 4 pounds of candy B
    produces 5 pounds of mixed candy that costs $2.00 per pound -/
theorem candy_mix_proof (candy_a_cost candy_b_cost mix_cost : ℝ)
                        (candy_a_weight candy_b_weight : ℝ) :
  candy_a_cost = 3.20 →
  candy_b_cost = 1.70 →
  mix_cost = 2.00 →
  candy_a_weight = 1 →
  candy_b_weight = 4 →
  (candy_a_cost * candy_a_weight + candy_b_cost * candy_b_weight) / 
    (candy_a_weight + candy_b_weight) = mix_cost :=
by sorry

end candy_mix_proof_l3291_329168


namespace garden_area_l3291_329185

/-- Given a square garden with perimeter 48 meters and a pond of area 20 square meters inside,
    the area of the garden not taken up by the pond is 124 square meters. -/
theorem garden_area (garden_perimeter : ℝ) (pond_area : ℝ) : 
  garden_perimeter = 48 → 
  pond_area = 20 → 
  (garden_perimeter / 4) ^ 2 - pond_area = 124 := by
  sorry

end garden_area_l3291_329185


namespace event_probability_l3291_329177

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by
sorry

end event_probability_l3291_329177


namespace sum_equals_222_l3291_329191

theorem sum_equals_222 : 148 + 35 + 17 + 13 + 9 = 222 := by
  sorry

end sum_equals_222_l3291_329191


namespace distance_case1_distance_case2_distance_formula_l3291_329165

-- Define a function to calculate the distance between two points on a number line
def distance (x1 x2 : ℝ) : ℝ := |x2 - x1|

-- Theorem for Case 1
theorem distance_case1 : distance 2 3 = 1 := by sorry

-- Theorem for Case 2
theorem distance_case2 : distance (-4) (-8) = 4 := by sorry

-- General theorem
theorem distance_formula (x1 x2 : ℝ) : 
  distance x1 x2 = |x2 - x1| := by sorry

end distance_case1_distance_case2_distance_formula_l3291_329165


namespace product_repeating_decimal_9_and_8_l3291_329182

/-- The repeating decimal 0.999... -/
def repeating_decimal_9 : ℝ := 0.999999

/-- Theorem: The product of 0.999... and 8 is equal to 8 -/
theorem product_repeating_decimal_9_and_8 : repeating_decimal_9 * 8 = 8 := by
  sorry

end product_repeating_decimal_9_and_8_l3291_329182


namespace correct_calculation_l3291_329146

theorem correct_calculation (x : ℚ) : x * 9 = 153 → x * 6 = 102 := by
  sorry

end correct_calculation_l3291_329146


namespace competition_results_l3291_329134

structure GradeData where
  boys_rate : ℝ
  girls_rate : ℝ

def seventh_grade : GradeData :=
  { boys_rate := 0.4, girls_rate := 0.6 }

def eighth_grade : GradeData :=
  { boys_rate := 0.5, girls_rate := 0.7 }

theorem competition_results :
  (seventh_grade.boys_rate < eighth_grade.boys_rate) ∧
  ((seventh_grade.boys_rate + eighth_grade.boys_rate) / 2 < (seventh_grade.girls_rate + eighth_grade.girls_rate) / 2) :=
by sorry

end competition_results_l3291_329134


namespace sandals_sold_l3291_329193

theorem sandals_sold (shoes : ℕ) (sandals : ℕ) : 
  (9 : ℚ) / 5 = shoes / sandals → shoes = 72 → sandals = 40 := by
  sorry

end sandals_sold_l3291_329193


namespace ceiling_floor_product_range_l3291_329117

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end ceiling_floor_product_range_l3291_329117


namespace jills_salary_l3291_329196

theorem jills_salary (discretionary_income : ℝ) (net_salary : ℝ) : 
  discretionary_income = net_salary / 5 →
  discretionary_income * 0.15 = 105 →
  net_salary = 3500 := by
  sorry

end jills_salary_l3291_329196


namespace curve_representation_l3291_329112

-- Define the equations
def equation1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0
def equation2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- Define what it means for an equation to represent a line and a circle
def represents_line_and_circle (f : ℝ → ℝ → Prop) : Prop :=
  (∃ a : ℝ, ∀ y, f a y) ∧ 
  (∃ c r : ℝ, ∀ x y, f x y ↔ (x - c)^2 + y^2 = r^2)

-- Define what it means for an equation to represent two points
def represents_two_points (f : ℝ → ℝ → Prop) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∨ y1 ≠ y2 ∧ 
    (∀ x y, f x y ↔ (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- State the theorem
theorem curve_representation :
  represents_line_and_circle equation1 ∧ 
  represents_two_points equation2 := by sorry

end curve_representation_l3291_329112


namespace min_a_for_polynomial_with_two_zeros_in_unit_interval_l3291_329148

theorem min_a_for_polynomial_with_two_zeros_in_unit_interval : 
  ∃ (a b c : ℤ), 
    (∀ (a' : ℤ), a' > 0 ∧ a' < a →
      ¬∃ (b' c' : ℤ), ∃ (x y : ℝ), 
        0 < x ∧ x < y ∧ y < 1 ∧
        a' * x^2 - b' * x + c' = 0 ∧
        a' * y^2 - b' * y + c' = 0) ∧
    (∃ (x y : ℝ), 
      0 < x ∧ x < y ∧ y < 1 ∧
      a * x^2 - b * x + c = 0 ∧
      a * y^2 - b * y + c = 0) ∧
    a = 5 := by sorry

end min_a_for_polynomial_with_two_zeros_in_unit_interval_l3291_329148


namespace g_definition_l3291_329176

-- Define the function f
def f (x : ℝ) : ℝ := 5 - 2*x

-- Define the function g
def g (x : ℝ) : ℝ := 4 - 3*x

-- Theorem statement
theorem g_definition (x : ℝ) : 
  (∀ y, f (y + 1) = 3 - 2*y) ∧ (f (g x) = 6*x - 3) → g x = 4 - 3*x :=
by
  sorry

end g_definition_l3291_329176


namespace simplify_expression_l3291_329198

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end simplify_expression_l3291_329198


namespace pascal_triangle_30_rows_sum_l3291_329184

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascalRowElements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascalTriangleSum (n : ℕ) : ℕ := 
  (n + 1) * (n + 2) / 2

theorem pascal_triangle_30_rows_sum :
  pascalTriangleSum 29 = 465 := by
  sorry

end pascal_triangle_30_rows_sum_l3291_329184


namespace sum_equals_fraction_l3291_329135

def binomial_coefficient (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def sum_expression : ℚ :=
  Finset.sum (Finset.range 8) (fun i =>
    let n := i + 3
    (binomial_coefficient n 2) / ((binomial_coefficient n 3) * (binomial_coefficient (n + 1) 3)))

theorem sum_equals_fraction :
  sum_expression = 164 / 165 :=
sorry

end sum_equals_fraction_l3291_329135


namespace clarinet_cost_calculation_l3291_329109

/-- The cost of items purchased at a music store -/
structure MusicStorePurchase where
  total_spent : ℝ
  songbook_cost : ℝ
  clarinet_cost : ℝ

/-- Theorem stating the cost of the clarinet given the total spent and songbook cost -/
theorem clarinet_cost_calculation (purchase : MusicStorePurchase) 
  (h1 : purchase.total_spent = 141.54)
  (h2 : purchase.songbook_cost = 11.24)
  : purchase.clarinet_cost = 130.30 := by
  sorry

end clarinet_cost_calculation_l3291_329109
