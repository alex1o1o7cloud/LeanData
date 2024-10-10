import Mathlib

namespace quadratic_solution_l3716_371648

theorem quadratic_solution (m : ℝ) : 
  (2 : ℝ)^2 - m * 2 + 8 = 0 → m = 6 := by
  sorry

end quadratic_solution_l3716_371648


namespace soup_cans_feeding_l3716_371691

/-- Proves that given 8 total cans of soup, where each can feeds either 4 adults or 6 children,
    and 18 children have been fed, the number of adults that can be fed with the remaining soup is 20. -/
theorem soup_cans_feeding (total_cans : ℕ) (adults_per_can children_per_can : ℕ) (children_fed : ℕ) :
  total_cans = 8 →
  adults_per_can = 4 →
  children_per_can = 6 →
  children_fed = 18 →
  (total_cans - (children_fed / children_per_can)) * adults_per_can = 20 :=
by sorry

end soup_cans_feeding_l3716_371691


namespace starting_number_of_range_l3716_371673

/-- Given a sequence of 10 consecutive multiples of 5 ending with 65,
    prove that the first number in the sequence is 15. -/
theorem starting_number_of_range (seq : Fin 10 → ℕ) : 
  (∀ i : Fin 10, seq i % 5 = 0) →  -- All numbers are divisible by 5
  (∀ i : Fin 9, seq i.succ = seq i + 5) →  -- Consecutive multiples of 5
  seq 9 = 65 →  -- The last number is 65
  seq 0 = 15 := by  -- The first number is 15
sorry


end starting_number_of_range_l3716_371673


namespace sequence_120th_term_l3716_371674

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth term of the sequence of positive integers 
    whose digits sum to 10, arranged in ascending order -/
def sequence_term (n : ℕ) : ℕ := sorry

/-- The main theorem: The 120th term of the sequence is 2017 -/
theorem sequence_120th_term : sequence_term 120 = 2017 := by sorry

end sequence_120th_term_l3716_371674


namespace rectangle_area_l3716_371638

/-- Given a rectangle with width w and length L, where L = w^2 and L + w = 25,
    prove that the area of the rectangle is (√101 - 1)^3 / 8 square inches. -/
theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) :
  w * L = ((Real.sqrt 101 - 1)^3) / 8 := by
  sorry

end rectangle_area_l3716_371638


namespace circle_center_sum_l3716_371617

theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 4*x + 10*y - 12 → x + y = 7 :=
by sorry

end circle_center_sum_l3716_371617


namespace smallest_gcd_qr_l3716_371699

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 240) (h2 : Nat.gcd p r = 540) :
  ∃ (q' r' : ℕ+), Nat.gcd q'.val r'.val = 60 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd q''.val r''.val ≥ 60 :=
sorry

end smallest_gcd_qr_l3716_371699


namespace not_all_tetrahedra_altitudes_intersect_l3716_371694

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- An altitude of a tetrahedron is a line segment from a vertex perpendicular to the opposite face -/
def Altitude (t : Tetrahedron) (v : Fin 4) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if all altitudes of a tetrahedron intersect at a single point -/
def altitudesIntersectAtPoint (t : Tetrahedron) : Prop :=
  ∃ p : ℝ × ℝ × ℝ, ∀ v : Fin 4, p ∈ Altitude t v

/-- Theorem stating that not all tetrahedra have altitudes intersecting at a single point -/
theorem not_all_tetrahedra_altitudes_intersect :
  ∃ t : Tetrahedron, ¬ altitudesIntersectAtPoint t :=
sorry

end not_all_tetrahedra_altitudes_intersect_l3716_371694


namespace largest_number_problem_l3716_371652

theorem largest_number_problem (a b c d : ℕ) 
  (sum_abc : a + b + c = 222)
  (sum_abd : a + b + d = 208)
  (sum_acd : a + c + d = 197)
  (sum_bcd : b + c + d = 180) :
  max a (max b (max c d)) = 89 := by
  sorry

end largest_number_problem_l3716_371652


namespace woodworker_tables_l3716_371618

theorem woodworker_tables (total_legs : ℕ) (chairs : ℕ) (chair_legs : ℕ) (table_legs : ℕ) 
  (h1 : total_legs = 40)
  (h2 : chairs = 6)
  (h3 : chair_legs = 4)
  (h4 : table_legs = 4) :
  (total_legs - chairs * chair_legs) / table_legs = 4 :=
by
  sorry

end woodworker_tables_l3716_371618


namespace gigi_jellybeans_l3716_371666

theorem gigi_jellybeans (gigi_jellybeans : ℕ) (rory_jellybeans : ℕ) (lorelai_jellybeans : ℕ) :
  rory_jellybeans = gigi_jellybeans + 30 →
  lorelai_jellybeans = 3 * (gigi_jellybeans + rory_jellybeans) →
  lorelai_jellybeans = 180 →
  gigi_jellybeans = 15 := by
sorry

end gigi_jellybeans_l3716_371666


namespace least_positive_linear_combination_l3716_371649

theorem least_positive_linear_combination :
  ∃ (n : ℕ), n > 0 ∧ (∀ (m : ℕ), m > 0 → (∃ (x y : ℤ), 24 * x + 20 * y = m) → m ≥ n) ∧
  (∃ (x y : ℤ), 24 * x + 20 * y = n) :=
by sorry

end least_positive_linear_combination_l3716_371649


namespace father_age_triple_weiwei_age_l3716_371631

/-- Weiwei's current age in years -/
def weiwei_age : ℕ := 8

/-- Weiwei's father's current age in years -/
def father_age : ℕ := 34

/-- The number of years after which the father's age will be three times Weiwei's age -/
def years_until_triple : ℕ := 5

theorem father_age_triple_weiwei_age :
  father_age + years_until_triple = 3 * (weiwei_age + years_until_triple) :=
sorry

end father_age_triple_weiwei_age_l3716_371631


namespace complex_equation_solution_l3716_371663

theorem complex_equation_solution (z : ℂ) : (Complex.I - z = 2 - Complex.I) → z = -2 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l3716_371663


namespace problem_1_l3716_371661

theorem problem_1 (a b : ℝ) : -2 * (a^2 - 4*b) + 3 * (2*a^2 - 4*b) = 4*a^2 - 4*b := by
  sorry

end problem_1_l3716_371661


namespace libby_igloo_bricks_l3716_371626

/-- Calculates the total number of bricks in an igloo -/
def igloo_bricks (total_rows : ℕ) (bottom_bricks_per_row : ℕ) (top_bricks_per_row : ℕ) : ℕ :=
  let bottom_rows := total_rows / 2
  let top_rows := total_rows - bottom_rows
  bottom_rows * bottom_bricks_per_row + top_rows * top_bricks_per_row

/-- Proves that Libby's igloo uses 100 bricks -/
theorem libby_igloo_bricks :
  igloo_bricks 10 12 8 = 100 := by
  sorry

end libby_igloo_bricks_l3716_371626


namespace august_math_problems_l3716_371698

theorem august_math_problems (a1 a2 a3 : ℕ) : 
  a1 = 600 →
  a3 = a1 + a2 - 400 →
  a1 + a2 + a3 = 3200 →
  a2 / a1 = 2 := by
sorry

end august_math_problems_l3716_371698


namespace two_distinct_roots_iff_a_in_open_interval_l3716_371622

-- Define the logarithmic function
noncomputable def log_base (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the main theorem
theorem two_distinct_roots_iff_a_in_open_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₁ + a > 0 ∧ x₁ + a ≠ 1 ∧
    x₂ > 0 ∧ x₂ + a > 0 ∧ x₂ + a ≠ 1 ∧
    log_base (x₁ + a) (2 * x₁) = 2 ∧
    log_base (x₂ + a) (2 * x₂) = 2) ↔
  (0 < a ∧ a < 1/2) :=
sorry

end two_distinct_roots_iff_a_in_open_interval_l3716_371622


namespace correct_calculation_l3716_371629

theorem correct_calculation (a : ℝ) : (-a + 3) * (-3 - a) = a^2 - 9 := by
  sorry

end correct_calculation_l3716_371629


namespace saras_golf_balls_l3716_371600

-- Define the number of dozens Sara has
def saras_dozens : ℕ := 9

-- Define the number of items in a dozen
def items_per_dozen : ℕ := 12

-- Theorem stating that Sara's total number of golf balls is 108
theorem saras_golf_balls : saras_dozens * items_per_dozen = 108 := by
  sorry

end saras_golf_balls_l3716_371600


namespace geometric_sequence_sum_l3716_371635

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 40 and a₃ + a₄ = 60, prove that a₇ + a₈ = 135 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {aₙ} is a geometric sequence
  a 1 + a 2 = 40 →                           -- a₁ + a₂ = 40
  a 3 + a 4 = 60 →                           -- a₃ + a₄ = 60
  a 7 + a 8 = 135 :=                         -- a₇ + a₈ = 135
by
  sorry

end geometric_sequence_sum_l3716_371635


namespace savings_calculation_l3716_371675

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income * 3 = expenditure * 5) →  -- Income and expenditure ratio is 5:3
  (income = 10000) →                -- Income is Rs. 10000
  (savings = income - expenditure) →  -- Definition of savings
  (savings = 4000) :=                -- Prove that savings are Rs. 4000
by
  sorry

#check savings_calculation

end savings_calculation_l3716_371675


namespace range_of_p_l3716_371695

def h (x : ℝ) : ℝ := 4 * x - 3

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ y ∈ Set.range (fun x => p x), 1 ≤ y ∧ y ≤ 129 ∧
  ∀ y, 1 ≤ y ∧ y ≤ 129 → ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ p x = y :=
by sorry

end range_of_p_l3716_371695


namespace product_inequality_find_a_l3716_371660

-- Part I
theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 + 1/a) * (1 + 1/b) ≥ 9 := by sorry

-- Part II
theorem find_a (a : ℝ) (h : ∀ x, |x + 3| - |x - a| ≥ 2 ↔ x ≥ 1) :
  a = 2 := by sorry

end product_inequality_find_a_l3716_371660


namespace expression_equality_l3716_371662

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 := by
  sorry

end expression_equality_l3716_371662


namespace min_weights_theorem_l3716_371687

/-- A function that calculates the sum of powers of 2 up to 2^n -/
def sumPowersOf2 (n : ℕ) : ℕ := 2^(n+1) - 1

/-- The maximum weight we need to measure -/
def maxWeight : ℕ := 100

/-- The proposition that n weights are sufficient to measure all weights up to maxWeight -/
def isSufficient (n : ℕ) : Prop := sumPowersOf2 n ≥ maxWeight

/-- The proposition that n weights are necessary to measure all weights up to maxWeight -/
def isNecessary (n : ℕ) : Prop := ∀ m : ℕ, m < n → sumPowersOf2 m < maxWeight

/-- The theorem stating that 7 is the minimum number of weights needed -/
theorem min_weights_theorem : 
  (isSufficient 7 ∧ isNecessary 7) ∧ ∀ n : ℕ, n < 7 → ¬(isSufficient n ∧ isNecessary n) :=
sorry

end min_weights_theorem_l3716_371687


namespace marked_elements_not_unique_l3716_371602

/-- Represents the table with 4 rows and 10 columns --/
def Table := Fin 4 → Fin 10 → Fin 10

/-- The table where each row is shifted by one position --/
def shiftedTable : Table :=
  λ i j => (j + i) % 10

/-- A marking of 10 elements in the table --/
def Marking := Fin 10 → Fin 4 × Fin 10

/-- Predicate to check if a marking is valid (one per row and column) --/
def isValidMarking (m : Marking) : Prop :=
  (∀ i : Fin 4, ∃! j : Fin 10, (i, j) ∈ Set.range m) ∧
  (∀ j : Fin 10, ∃! i : Fin 4, (i, j) ∈ Set.range m)

theorem marked_elements_not_unique (t : Table) (m : Marking) 
  (h : isValidMarking m) : 
  ∃ i j : Fin 10, i ≠ j ∧ t (m i).1 (m i).2 = t (m j).1 (m j).2 :=
sorry

end marked_elements_not_unique_l3716_371602


namespace quadratic_inequality_solution_set_l3716_371609

theorem quadratic_inequality_solution_set (x : ℝ) : x^2 + 3*x - 4 < 0 ↔ -4 < x ∧ x < 1 := by
  sorry

end quadratic_inequality_solution_set_l3716_371609


namespace victoria_friends_l3716_371624

theorem victoria_friends (total_pairs : ℕ) (shoes_per_person : ℕ) (victoria_shoes : ℕ) : 
  total_pairs = 36 →
  shoes_per_person = 2 →
  victoria_shoes = 2 →
  (total_pairs * 2 - victoria_shoes) % shoes_per_person = 0 →
  (total_pairs * 2 - victoria_shoes) / shoes_per_person = 35 := by
  sorry

end victoria_friends_l3716_371624


namespace growth_rate_correct_l3716_371692

/-- The average annual growth rate of vegetable production value from 2013 to 2015 -/
def average_growth_rate : ℝ := 0.25

/-- The initial production value in 2013 (in millions of yuan) -/
def initial_value : ℝ := 6.4

/-- The final production value in 2015 (in millions of yuan) -/
def final_value : ℝ := 10

/-- Theorem stating that the average annual growth rate correctly relates the initial and final values -/
theorem growth_rate_correct : initial_value * (1 + average_growth_rate)^2 = final_value := by
  sorry

end growth_rate_correct_l3716_371692


namespace no_three_similar_piles_l3716_371634

theorem no_three_similar_piles : ¬∃ (x a b c : ℝ), 
  (x > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (a + b + c = x) ∧
  (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
  (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
  (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) := by
  sorry

end no_three_similar_piles_l3716_371634


namespace three_in_all_curriculums_l3716_371655

/-- Represents the number of people in different curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  cookingAndWeaving : ℕ

/-- Calculates the number of people participating in all curriculums -/
def allCurriculums (g : CurriculumGroups) : ℕ :=
  g.cooking - g.cookingOnly - g.cookingAndYoga - g.cookingAndWeaving

/-- Theorem stating that 3 people participate in all curriculums -/
theorem three_in_all_curriculums (g : CurriculumGroups) 
  (h1 : g.yoga = 35)
  (h2 : g.cooking = 20)
  (h3 : g.weaving = 15)
  (h4 : g.cookingOnly = 7)
  (h5 : g.cookingAndYoga = 5)
  (h6 : g.cookingAndWeaving = 5) :
  allCurriculums g = 3 := by
  sorry

end three_in_all_curriculums_l3716_371655


namespace quadratic_inequality_solution_set_l3716_371650

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) →
  b^2 - 4*a*c < 0 ∧
  ¬(b^2 - 4*a*c < 0 → ∀ x : ℝ, a * x^2 + b * x + c > 0) :=
by sorry

end quadratic_inequality_solution_set_l3716_371650


namespace quadratic_one_root_l3716_371647

theorem quadratic_one_root (c b : ℝ) (hc : c > 0) :
  (∃! x : ℝ, x^2 + 2 * Real.sqrt c * x + b = 0) → c = b := by
  sorry

end quadratic_one_root_l3716_371647


namespace income_comparison_l3716_371637

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mart = tim * (1 + 0.4)) :
  mart = juan * 0.84 := by
sorry

end income_comparison_l3716_371637


namespace intersection_implies_a_value_l3716_371646

def M (a : ℤ) : Set ℤ := {a, 0}

def N : Set ℤ := {x : ℤ | x^2 - 3*x < 0}

theorem intersection_implies_a_value (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := by
  sorry

end intersection_implies_a_value_l3716_371646


namespace f_properties_l3716_371628

noncomputable def f (x : ℝ) := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 2) ∧
  f (3 * Real.pi / 8) = -Real.sqrt 2 := by
  sorry

end f_properties_l3716_371628


namespace tangent_line_at_2_a_range_l3716_371639

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m*x + b ↔ y = 8*x - 2 ∧ 
    (∃ (h : ℝ), h ≠ 0 ∧ (f 1 (2 + h) - f 1 2) / h = m) :=
sorry

-- Part 2: Range of a
theorem a_range :
  ∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f a x < 0) ↔ a > 9/2 :=
sorry

end tangent_line_at_2_a_range_l3716_371639


namespace original_number_proof_l3716_371607

theorem original_number_proof : ∃ N : ℕ, N > 0 ∧ N - 28 ≡ 0 [MOD 87] ∧ ∀ M : ℕ, M > 0 ∧ M - 28 ≡ 0 [MOD 87] → M ≥ N :=
by
  -- The proof goes here
  sorry

end original_number_proof_l3716_371607


namespace function_property_l3716_371623

theorem function_property (f : ℤ → ℤ) 
  (h : ∀ (a b : ℤ), a ≠ 0 → b ≠ 0 → f (a * b) ≥ f a + f b) :
  ∀ (a : ℤ), a ≠ 0 → (∀ (n : ℕ), f (a ^ n) = n * f a) ↔ f (a ^ 2) = 2 * f a :=
sorry

end function_property_l3716_371623


namespace sum_of_multiples_plus_eleven_l3716_371633

theorem sum_of_multiples_plus_eleven : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end sum_of_multiples_plus_eleven_l3716_371633


namespace class_size_is_40_l3716_371678

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat
  threeOrMore : Nat

/-- Calculates the total number of students given the book borrowing data -/
def totalStudents (b : BookBorrowers) : Nat :=
  b.zero + b.one + b.two + b.threeOrMore

/-- Calculates the minimum number of books borrowed -/
def minBooksBorrowed (b : BookBorrowers) : Nat :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- The given book borrowing data for the class -/
def classBorrowers : BookBorrowers := {
  zero := 2,
  one := 12,
  two := 10,
  threeOrMore := 16  -- This value is not given directly, but can be derived
}

theorem class_size_is_40 :
  totalStudents classBorrowers = 40 ∧
  (minBooksBorrowed classBorrowers : ℚ) / (totalStudents classBorrowers) = 2 := by
  sorry


end class_size_is_40_l3716_371678


namespace broken_flagpole_l3716_371690

theorem broken_flagpole (h : ℝ) (d : ℝ) (x : ℝ) 
  (height_cond : h = 10)
  (distance_cond : d = 4) :
  (x^2 + d^2 = (h - x)^2) → x = 2 * Real.sqrt 22 :=
by sorry

end broken_flagpole_l3716_371690


namespace floor_sqrt_80_l3716_371601

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end floor_sqrt_80_l3716_371601


namespace cos_pi_minus_theta_l3716_371612

theorem cos_pi_minus_theta (θ : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.cos (π - θ) = -4/5 := by
  sorry

end cos_pi_minus_theta_l3716_371612


namespace similar_triangles_side_length_l3716_371697

theorem similar_triangles_side_length 
  (A₁ A₂ : ℕ) (k : ℕ) (side_small : ℝ) :
  A₁ > A₂ →
  A₁ - A₂ = 32 →
  A₁ = k^2 * A₂ →
  side_small = 4 →
  ∃ (side_large : ℝ), side_large = 12 :=
by sorry

end similar_triangles_side_length_l3716_371697


namespace tom_apple_purchase_l3716_371640

-- Define the given constants
def apple_price : ℝ := 70
def mango_amount : ℝ := 9
def mango_price : ℝ := 65
def total_paid : ℝ := 1145

-- Define the theorem
theorem tom_apple_purchase :
  ∃ (apple_amount : ℝ),
    apple_amount * apple_price + mango_amount * mango_price = total_paid ∧
    apple_amount = 8 := by
  sorry

end tom_apple_purchase_l3716_371640


namespace smallest_number_divisible_by_three_prime_squares_l3716_371682

def is_divisible_by_three_prime_squares (n : ℕ) : Prop :=
  ∃ p q r : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    n % (p^2) = 0 ∧ n % (q^2) = 0 ∧ n % (r^2) = 0

theorem smallest_number_divisible_by_three_prime_squares :
  (∀ m : ℕ, m > 0 ∧ m < 900 → ¬(is_divisible_by_three_prime_squares m)) ∧
  is_divisible_by_three_prime_squares 900 := by
  sorry

end smallest_number_divisible_by_three_prime_squares_l3716_371682


namespace plot_area_in_acres_l3716_371616

/-- Conversion factor from square miles to acres -/
def miles_to_acres : ℝ := 640

/-- Length of the plot in miles -/
def length : ℝ := 12

/-- Width of the plot in miles -/
def width : ℝ := 8

/-- Theorem stating that the area of the rectangular plot in acres is 61440 -/
theorem plot_area_in_acres :
  length * width * miles_to_acres = 61440 := by sorry

end plot_area_in_acres_l3716_371616


namespace sandwich_toppings_combinations_l3716_371643

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem sandwich_toppings_combinations :
  choose 9 3 = 84 := by sorry

end sandwich_toppings_combinations_l3716_371643


namespace negation_of_universal_proposition_l3716_371619

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l3716_371619


namespace min_garden_cost_l3716_371658

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower with its price -/
structure Flower where
  price : ℝ

/-- The garden layout -/
def garden : List Region := [
  ⟨5, 2⟩, -- Region 1
  ⟨3, 5⟩, -- Region 2
  ⟨2, 4⟩, -- Region 3
  ⟨5, 4⟩, -- Region 4
  ⟨5, 3⟩  -- Region 5
]

/-- Available flowers with their prices -/
def flowers : List Flower := [
  ⟨1.20⟩, -- Asters
  ⟨1.70⟩, -- Begonias
  ⟨2.20⟩, -- Cannas
  ⟨2.70⟩, -- Dahlias
  ⟨3.20⟩  -- Freesias
]

/-- Calculate the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculate the total area of the garden -/
def totalArea : ℝ := List.sum (List.map area garden)

/-- The main theorem: prove that the minimum cost is $152.60 -/
theorem min_garden_cost :
  ∃ (assignment : List (Region × Flower)),
    (List.length assignment = List.length garden) ∧
    (∀ r ∈ garden, ∃ f ∈ flowers, (r, f) ∈ assignment) ∧
    (List.sum (List.map (λ (r, f) => area r * f.price) assignment) = 152.60) ∧
    (∀ other_assignment : List (Region × Flower),
      (List.length other_assignment = List.length garden) →
      (∀ r ∈ garden, ∃ f ∈ flowers, (r, f) ∈ other_assignment) →
      List.sum (List.map (λ (r, f) => area r * f.price) other_assignment) ≥ 152.60) :=
by sorry

end min_garden_cost_l3716_371658


namespace polygon_diagonal_division_l3716_371651

/-- 
For an n-sided polygon, if a diagonal drawn from a vertex can divide it into 
at most 2023 triangles, then n = 2025.
-/
theorem polygon_diagonal_division (n : ℕ) : 
  (∃ (d : ℕ), d ≤ 2023 ∧ d = n - 2) → n = 2025 := by
  sorry

end polygon_diagonal_division_l3716_371651


namespace smallest_scalene_triangle_perimeter_l3716_371625

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a valid triangle -/
def isTriangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  a < b ∧ b < c ∧
  b = a + 2 ∧ c = b + 2

theorem smallest_scalene_triangle_perimeter :
  ∀ p q r : ℕ,
    areConsecutiveOddPrimes p q r →
    isTriangle p q r →
    isPrime (p + q + r) →
    p + q + r ≥ 23 :=
sorry

end smallest_scalene_triangle_perimeter_l3716_371625


namespace one_root_in_interval_l3716_371605

theorem one_root_in_interval : ∃! x : ℝ, 0 < x ∧ x < 2 ∧ 2 * x^3 - 6 * x^2 + 7 = 0 := by
  sorry

end one_root_in_interval_l3716_371605


namespace f_max_value_l3716_371659

/-- The quadratic function f(x) = -5x^2 + 25x - 1 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 1

/-- The maximum value of f(x) is 129/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 129 / 4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end f_max_value_l3716_371659


namespace grid_game_winner_l3716_371672

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | Player1Wins
  | Player2Wins

/-- Represents the game state on a 1 × N grid strip -/
structure GameState (N : ℕ) where
  grid : Fin N → Option Bool
  turn : Bool

/-- Defines the game rules and winning conditions -/
def gameResult (N : ℕ) : GameOutcome :=
  if N = 1 then
    GameOutcome.Player1Wins
  else
    GameOutcome.Player2Wins

/-- Theorem stating the winning player based on the grid size -/
theorem grid_game_winner (N : ℕ) :
  (N = 1 → gameResult N = GameOutcome.Player1Wins) ∧
  (N > 1 → gameResult N = GameOutcome.Player2Wins) := by
  sorry

/-- Lemma: Player 1 wins when N = 1 -/
lemma player1_wins_n1 (N : ℕ) (h : N = 1) :
  gameResult N = GameOutcome.Player1Wins := by
  sorry

/-- Lemma: Player 2 wins when N > 1 -/
lemma player2_wins_n_gt1 (N : ℕ) (h : N > 1) :
  gameResult N = GameOutcome.Player2Wins := by
  sorry

end grid_game_winner_l3716_371672


namespace tree_distance_l3716_371696

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 6) (h2 : d = 60) :
  (n - 1) * (d / 3) = 100 := by
  sorry

end tree_distance_l3716_371696


namespace mothers_age_l3716_371684

theorem mothers_age (daughter_age mother_age : ℕ) 
  (h1 : 2 * daughter_age + mother_age = 70)
  (h2 : daughter_age + 2 * mother_age = 95) :
  mother_age = 40 := by
  sorry

end mothers_age_l3716_371684


namespace unread_fraction_of_book_l3716_371677

theorem unread_fraction_of_book (total : ℝ) (read : ℝ) : 
  total > 0 → read > total / 2 → read < total → (total - read) / total = 2 / 5 := by
  sorry

end unread_fraction_of_book_l3716_371677


namespace symmetry_sum_for_17gon_l3716_371669

/-- The number of sides in our regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle of rotational symmetry (in degrees) for a regular n-gon -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 649/17 -/
theorem symmetry_sum_for_17gon : L n + R n = 649 / 17 := by
  sorry

end symmetry_sum_for_17gon_l3716_371669


namespace token_game_ends_in_37_rounds_l3716_371668

/-- Represents the state of the game at any given round -/
structure GameState where
  tokensA : ℕ
  tokensB : ℕ
  tokensC : ℕ

/-- Represents the rules of the game -/
def nextRound (state : GameState) : GameState :=
  match state with
  | ⟨a, b, c⟩ =>
    if a ≥ b ∧ a ≥ c then ⟨a - 3, b + 1, c + 1⟩
    else if b ≥ a ∧ b ≥ c then ⟨a + 1, b - 3, c + 1⟩
    else ⟨a + 1, b + 1, c - 3⟩

/-- Checks if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  state.tokensA = 0 ∨ state.tokensB = 0 ∨ state.tokensC = 0

/-- Plays the game for a given number of rounds -/
def playGame (initialState : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initialState
  | n + 1 => nextRound (playGame initialState n)

/-- The main theorem statement -/
theorem token_game_ends_in_37_rounds :
  let initialState := GameState.mk 15 14 13
  gameEnded (playGame initialState 37) ∧ ¬gameEnded (playGame initialState 36) := by
  sorry


end token_game_ends_in_37_rounds_l3716_371668


namespace cube_surface_area_l3716_371685

/-- The surface area of a cube with edge length 7 cm is 294 square centimeters. -/
theorem cube_surface_area : 
  ∀ (edge_length : ℝ), 
  edge_length = 7 → 
  6 * edge_length^2 = 294 := by
  sorry

end cube_surface_area_l3716_371685


namespace overlap_length_l3716_371664

theorem overlap_length (total_length actual_distance : ℝ) (num_overlaps : ℕ) : 
  total_length = 98 → 
  actual_distance = 83 → 
  num_overlaps = 6 → 
  (total_length - actual_distance) / num_overlaps = 2.5 := by
  sorry

end overlap_length_l3716_371664


namespace cubic_quadratic_fraction_inequality_l3716_371679

theorem cubic_quadratic_fraction_inequality (s r : ℝ) (hs : 0 < s) (hr : 0 < r) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end cubic_quadratic_fraction_inequality_l3716_371679


namespace nested_f_result_l3716_371621

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem nested_f_result (p q : ℝ) :
  (∀ x ∈ Set.Icc 1 3, |f p q x| ≤ 1/2) →
  (f p q)^[2017] ((3 + Real.sqrt 7) / 2) = (3 - Real.sqrt 7) / 2 :=
by sorry

end nested_f_result_l3716_371621


namespace erdos_binomial_prime_factors_l3716_371670

-- Define the number of distinct prime factors function
noncomputable def num_distinct_prime_factors (m : ℕ) : ℕ := sorry

-- State the theorem
theorem erdos_binomial_prime_factors :
  ∃ (c : ℝ), c > 1 ∧
  ∀ (n k : ℕ), n > 0 ∧ k > 0 →
  (n : ℝ) > c^k →
  num_distinct_prime_factors (Nat.choose n k) ≥ k :=
sorry

end erdos_binomial_prime_factors_l3716_371670


namespace scientific_notation_of_32_9_billion_l3716_371613

def billion : ℝ := 1000000000

theorem scientific_notation_of_32_9_billion :
  32.9 * billion = 3.29 * (10 : ℝ)^9 := by sorry

end scientific_notation_of_32_9_billion_l3716_371613


namespace polygon_sides_count_l3716_371689

/-- Represents the number of degrees in a circle -/
def degrees_in_circle : ℝ := 360

/-- Represents the common difference in the arithmetic progression of angles -/
def common_difference : ℝ := 3

/-- Represents the measure of the largest angle in the polygon -/
def largest_angle : ℝ := 150

/-- Theorem: A convex polygon with interior angles in arithmetic progression,
    a common difference of 3°, and the largest angle of 150° has 48 sides -/
theorem polygon_sides_count :
  ∀ n : ℕ,
  (n > 2) →
  (n * (2 * largest_angle - (n - 1) * common_difference) / 2 = (n - 2) * degrees_in_circle / 2) →
  n = 48 := by
  sorry


end polygon_sides_count_l3716_371689


namespace exponent_addition_l3716_371632

theorem exponent_addition (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_addition_l3716_371632


namespace sqrt_98_plus_sqrt_32_l3716_371671

theorem sqrt_98_plus_sqrt_32 : Real.sqrt 98 + Real.sqrt 32 = 11 * Real.sqrt 2 := by
  sorry

end sqrt_98_plus_sqrt_32_l3716_371671


namespace perpendicular_transitivity_l3716_371627

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the different relation
variable (different : ∀ {α : Type}, α → α → Prop)

theorem perpendicular_transitivity 
  (α β γ : Plane) (m n l : Line)
  (h_diff_planes : different α β ∧ different β γ ∧ different α γ)
  (h_diff_lines : different m n ∧ different n l ∧ different m l)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end perpendicular_transitivity_l3716_371627


namespace binomial_prob_one_third_l3716_371620

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_prob_one_third 
  (X : BinomialRV) 
  (h_expect : expectation X = 30)
  (h_var : variance X = 20) : 
  X.p = 1/3 := by
sorry

end binomial_prob_one_third_l3716_371620


namespace incorrect_observation_value_l3716_371657

/-- Given a set of observations with known properties, determine the value of an incorrectly recorded observation. -/
theorem incorrect_observation_value
  (n : ℕ)  -- Total number of observations
  (original_mean : ℝ)  -- Original mean of all observations
  (correct_value : ℝ)  -- The correct value of the misrecorded observation
  (new_mean : ℝ)  -- The new mean after correcting the misrecorded observation
  (h_n : n = 50)  -- There are 50 observations
  (h_original_mean : original_mean = 36)  -- The original mean was 36
  (h_correct_value : correct_value = 30)  -- The correct value should have been 30
  (h_new_mean : new_mean = 36.5)  -- The new mean after correction is 36.5
  : ∃ (incorrect_value : ℝ), incorrect_value = 55 := by
  sorry

end incorrect_observation_value_l3716_371657


namespace smallest_non_odd_unit_digit_l3716_371630

def OddUnitDigits : Set ℕ := {1, 3, 5, 7, 9}

def IsOdd (n : ℕ) : Prop := n % 2 = 1

def UnitsDigit (n : ℕ) : ℕ := n % 10

theorem smallest_non_odd_unit_digit :
  (∀ n : ℕ, IsOdd n → UnitsDigit n ∈ OddUnitDigits) →
  (∀ d : ℕ, d < 0 → d ∉ OddUnitDigits) →
  (∀ d : ℕ, 0 < d → d < 10 → d ∉ OddUnitDigits → 0 < d) →
  (0 ∉ OddUnitDigits ∧ ∀ d : ℕ, d < 10 → d ∉ OddUnitDigits → 0 ≤ d) :=
by sorry

end smallest_non_odd_unit_digit_l3716_371630


namespace laptop_selection_problem_l3716_371686

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem laptop_selection_problem :
  let type_a : ℕ := 4
  let type_b : ℕ := 5
  let total_selection : ℕ := 3
  (choose type_a 2 * choose type_b 1) + (choose type_a 1 * choose type_b 2) = 70 :=
by sorry

end laptop_selection_problem_l3716_371686


namespace expression_evaluation_l3716_371641

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - y^2 ≠ 0) :
  (y^2 - 1/x) / (x - y^2) = (x*y^2 - 1) / (x^2 - x*y^2) := by
  sorry

end expression_evaluation_l3716_371641


namespace mph_to_fps_conversion_l3716_371656

/-- Conversion factor from miles per hour to feet per second -/
def mph_to_fps : ℝ := 1.5

/-- Cheetah's speed in miles per hour -/
def cheetah_speed : ℝ := 60

/-- Gazelle's speed in miles per hour -/
def gazelle_speed : ℝ := 40

/-- Initial distance between cheetah and gazelle in feet -/
def initial_distance : ℝ := 210

/-- Time for cheetah to catch up to gazelle in seconds -/
def catch_up_time : ℝ := 7

theorem mph_to_fps_conversion :
  (cheetah_speed * mph_to_fps * catch_up_time) - (gazelle_speed * mph_to_fps * catch_up_time) = initial_distance := by
  sorry

#check mph_to_fps_conversion

end mph_to_fps_conversion_l3716_371656


namespace stripe_area_on_cylindrical_silo_l3716_371603

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h_diameter : diameter = 20) 
  (h_stripe_width : stripe_width = 4) 
  (h_revolutions : revolutions = 4) : 
  stripe_width * revolutions * (π * diameter) = 640 * π := by
sorry

end stripe_area_on_cylindrical_silo_l3716_371603


namespace integral_abs_x_squared_minus_x_l3716_371642

theorem integral_abs_x_squared_minus_x : ∫ x in (-1)..1, |x^2 - x| = 1/3 := by
  sorry

end integral_abs_x_squared_minus_x_l3716_371642


namespace smallest_positive_integer_with_given_remainders_l3716_371654

theorem smallest_positive_integer_with_given_remainders :
  ∃ b : ℕ, b > 0 ∧ 
    b % 5 = 4 ∧ 
    b % 7 = 6 ∧ 
    b % 11 = 10 ∧ 
    (∀ c : ℕ, c > 0 ∧ c % 5 = 4 ∧ c % 7 = 6 ∧ c % 11 = 10 → b ≤ c) ∧
    b = 384 := by
  sorry

end smallest_positive_integer_with_given_remainders_l3716_371654


namespace brave_2022_first_appearance_l3716_371608

/-- The cycle length of the letters "BRAVE" -/
def letter_cycle_length : ℕ := 5

/-- The cycle length of the digits "2022" -/
def digit_cycle_length : ℕ := 4

/-- The line number where "BRAVE 2022" first appears -/
def first_appearance : ℕ := 20

theorem brave_2022_first_appearance :
  Nat.lcm letter_cycle_length digit_cycle_length = first_appearance :=
by sorry

end brave_2022_first_appearance_l3716_371608


namespace f_neg_two_value_l3716_371604

/-- Given a function f(x) = -ax^5 - x^3 + bx - 7, if f(2) = -9, then f(-2) = -5 -/
theorem f_neg_two_value (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ -a * x^5 - x^3 + b * x - 7
  f 2 = -9 → f (-2) = -5 := by
sorry

end f_neg_two_value_l3716_371604


namespace number_ratio_l3716_371645

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y = 3 * x) (h4 : x + y = 124) :
  x / y = 1 / 3 := by
sorry

end number_ratio_l3716_371645


namespace six_digit_divisible_by_7_8_9_l3716_371614

theorem six_digit_divisible_by_7_8_9 :
  ∃ n : ℕ, 523000 ≤ n ∧ n ≤ 523999 ∧ 7 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n := by
  sorry

end six_digit_divisible_by_7_8_9_l3716_371614


namespace ball_color_probability_l3716_371676

theorem ball_color_probability : 
  let n : ℕ := 8
  let p : ℝ := 1/2
  let num_arrangements : ℕ := n.choose (n/2)
  Fintype.card {s : Finset (Fin n) | s.card = n/2} / 2^n = 35/128 :=
by sorry

end ball_color_probability_l3716_371676


namespace farmer_animals_l3716_371688

theorem farmer_animals (cows pigs goats chickens ducks sheep : ℕ) : 
  pigs = 3 * cows →
  cows = goats + 7 →
  chickens = 2 * (cows + pigs) →
  2 * ducks = goats + chickens →
  sheep = cows + chickens + 5 →
  cows + pigs + goats + chickens + ducks + sheep = 346 →
  goats = 6 := by
sorry

end farmer_animals_l3716_371688


namespace unique_r_value_l3716_371680

/-- The polynomial function f(x) -/
def f (r : ℝ) (x : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 5 * x + r

/-- Theorem stating that r = -5 is the unique value that satisfies f(-1) = 0 -/
theorem unique_r_value : ∃! r : ℝ, f r (-1) = 0 ∧ r = -5 := by sorry

end unique_r_value_l3716_371680


namespace lcm_18_24_l3716_371610

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l3716_371610


namespace total_marks_calculation_l3716_371611

theorem total_marks_calculation (num_candidates : ℕ) (average_marks : ℕ) 
  (h1 : num_candidates = 120) (h2 : average_marks = 35) : 
  num_candidates * average_marks = 4200 := by
  sorry

end total_marks_calculation_l3716_371611


namespace rita_swim_hours_l3716_371667

/-- The total number of hours Rita needs to swim --/
def total_swim_hours (backstroke breaststroke butterfly monthly_freestyle_sidestroke months : ℕ) : ℕ :=
  backstroke + breaststroke + butterfly + monthly_freestyle_sidestroke * months

/-- Theorem stating that Rita needs to swim 1500 hours in total --/
theorem rita_swim_hours :
  total_swim_hours 50 9 121 220 6 = 1500 :=
by sorry

end rita_swim_hours_l3716_371667


namespace two_tvs_one_mixer_cost_l3716_371636

/-- The cost of a mixer in rupees -/
def mixer_cost : ℕ := 1400

/-- The cost of a TV in rupees -/
def tv_cost : ℕ := 4200

/-- The cost of two mixers and one TV in rupees -/
def two_mixers_one_tv_cost : ℕ := 7000

theorem two_tvs_one_mixer_cost : 2 * tv_cost + mixer_cost = 9800 := by
  sorry

end two_tvs_one_mixer_cost_l3716_371636


namespace seojun_apple_fraction_l3716_371681

theorem seojun_apple_fraction :
  let total_apples : ℕ := 100
  let seojun_apples : ℕ := 11
  (seojun_apples : ℚ) / total_apples = 0.11 := by
  sorry

end seojun_apple_fraction_l3716_371681


namespace max_k_value_l3716_371653

theorem max_k_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → (x + 2*y) / (x*y) ≥ k / (2*x + y)) →
  k ≤ 9 :=
by sorry

end max_k_value_l3716_371653


namespace final_temperature_l3716_371683

def initial_temp : Int := -3
def temp_rise : Int := 6
def temp_drop : Int := 7

theorem final_temperature : 
  initial_temp + temp_rise - temp_drop = -4 :=
by sorry

end final_temperature_l3716_371683


namespace quadratic_inequality_solution_set_l3716_371644

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end quadratic_inequality_solution_set_l3716_371644


namespace lt_iff_forall_add_lt_l3716_371693

theorem lt_iff_forall_add_lt (a b : ℝ) : a < b ↔ ∀ x ∈ Set.Ioo 0 1, a + x < b := by sorry

end lt_iff_forall_add_lt_l3716_371693


namespace open_box_volume_l3716_371665

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 5) :
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 9880 :=
by sorry

end open_box_volume_l3716_371665


namespace john_works_five_days_l3716_371606

/-- Represents the number of widgets John can make per hour -/
def widgets_per_hour : ℕ := 20

/-- Represents the number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- Represents the total number of widgets John makes per week -/
def widgets_per_week : ℕ := 800

/-- Calculates the number of days John works per week -/
def days_worked_per_week : ℕ :=
  widgets_per_week /(widgets_per_hour * hours_per_day)

/-- Theorem stating that John works 5 days per week -/
theorem john_works_five_days :
  days_worked_per_week = 5 := by
  sorry

end john_works_five_days_l3716_371606


namespace intersection_points_form_line_l3716_371615

theorem intersection_points_form_line (s : ℝ) : 
  ∃ (x y : ℝ), 2*x + 3*y = 8*s + 4 ∧ 3*x - 4*y = 9*s - 3 → 
  y = (20/59)*x + 60/59 :=
sorry

end intersection_points_form_line_l3716_371615
