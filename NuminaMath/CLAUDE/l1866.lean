import Mathlib

namespace continued_fraction_solution_l1866_186655

theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) ∧ x = (3 + Real.sqrt 69) / 2 := by
  sorry

end continued_fraction_solution_l1866_186655


namespace technicians_schedule_lcm_l1866_186648

theorem technicians_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end technicians_schedule_lcm_l1866_186648


namespace rational_function_equality_l1866_186640

theorem rational_function_equality (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 120*x + 3480) / (x^2 + 54*x - 2835)) →
  α + β = 123 := by
sorry

end rational_function_equality_l1866_186640


namespace valid_triangle_l1866_186642

/-- A triangle with side lengths a, b, and c satisfies the triangle inequality theorem -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of numbers (2, 3, 4) forms a valid triangle -/
theorem valid_triangle : is_triangle 2 3 4 := by
  sorry

end valid_triangle_l1866_186642


namespace special_sequence_bound_l1866_186619

def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k : ℕ, k ∈ Set.range a ∨ ∃ i j, k = a i + a j)

theorem special_sequence_bound (a : ℕ → ℕ) (h : SpecialSequence a) : 
  ∀ n : ℕ, n > 0 → a n ≤ n^2 := by
sorry

end special_sequence_bound_l1866_186619


namespace square_side_bounds_l1866_186643

/-- A triangle with an inscribed square and circle -/
structure TriangleWithInscriptions where
  /-- The side length of the inscribed square -/
  s : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The square is inscribed such that two vertices are on the base and two on the sides -/
  square_inscribed : True
  /-- The circle is inscribed in the triangle -/
  circle_inscribed : True
  /-- Both s and r are positive -/
  s_pos : 0 < s
  r_pos : 0 < r

/-- The side of the inscribed square is bounded by √2r and 2r -/
theorem square_side_bounds (t : TriangleWithInscriptions) : Real.sqrt 2 * t.r < t.s ∧ t.s < 2 * t.r := by
  sorry

end square_side_bounds_l1866_186643


namespace m_intersect_n_equals_n_l1866_186666

open Set

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def N : Set ℝ := {x : ℝ | x > 3}

-- Theorem statement
theorem m_intersect_n_equals_n : M ∩ N = N := by
  sorry

end m_intersect_n_equals_n_l1866_186666


namespace polynomial_factorization_l1866_186630

theorem polynomial_factorization (a : ℝ) : a^3 + a^2 - a - 1 = (a - 1) * (a + 1)^2 := by
  sorry

end polynomial_factorization_l1866_186630


namespace tangent_line_at_M_l1866_186622

/-- The circle with equation x^2 + y^2 = 5 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 5}

/-- The point M on the circle -/
def M : ℝ × ℝ := (2, -1)

/-- The proposed tangent line equation -/
def TangentLine (x y : ℝ) : Prop := 2*x - y - 5 = 0

/-- Theorem stating that the proposed line is tangent to the circle at M -/
theorem tangent_line_at_M :
  M ∈ Circle ∧
  TangentLine M.1 M.2 ∧
  ∀ p ∈ Circle, p ≠ M → ¬TangentLine p.1 p.2 :=
sorry

end tangent_line_at_M_l1866_186622


namespace parallelogram_area_l1866_186675

/-- The area of a parallelogram with base 22 cm and height 21 cm is 462 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 22 → 
    height = 21 → 
    area = base * height → 
    area = 462 := by
  sorry

end parallelogram_area_l1866_186675


namespace triangle_BC_length_l1866_186624

/-- Triangle ABC with given properties --/
structure TriangleABC where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BX : ℕ
  CX : ℕ
  h_AB : AB = 75
  h_AC : AC = 85
  h_BC : BC = BX + CX
  h_circle : BX^2 + CX^2 = AB^2

/-- Theorem: BC = 89 in the given triangle --/
theorem triangle_BC_length (t : TriangleABC) : t.BC = 89 := by
  sorry

end triangle_BC_length_l1866_186624


namespace cos_54_degrees_l1866_186664

theorem cos_54_degrees : Real.cos (54 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end cos_54_degrees_l1866_186664


namespace vector_operation_l1866_186638

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (2, -2)) :
  2 • a - b = (4, 4) := by
  sorry

end vector_operation_l1866_186638


namespace expression_simplification_l1866_186694

theorem expression_simplification :
  4 * Real.sqrt 2 * Real.sqrt 3 - Real.sqrt 12 / Real.sqrt 2 + Real.sqrt 24 = 5 * Real.sqrt 6 := by
  sorry

end expression_simplification_l1866_186694


namespace evening_painting_l1866_186658

/-- A dodecahedron is a polyhedron with 12 faces -/
def dodecahedron_faces : ℕ := 12

/-- The number of faces Samuel painted in the morning -/
def painted_faces : ℕ := 5

/-- The number of faces Samuel needs to paint in the evening -/
def remaining_faces : ℕ := dodecahedron_faces - painted_faces

theorem evening_painting : remaining_faces = 7 := by
  sorry

end evening_painting_l1866_186658


namespace solution_difference_l1866_186626

theorem solution_difference (p q : ℝ) : 
  (p - 3) * (p + 3) = 21 * p - 63 →
  (q - 3) * (q + 3) = 21 * q - 63 →
  p ≠ q →
  p > q →
  p - q = 15 := by
sorry

end solution_difference_l1866_186626


namespace answer_key_combinations_l1866_186678

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- Represents the total number of possible true-false combinations -/
def total_true_false_combinations : ℕ := 2^true_false_questions

/-- Represents the number of invalid true-false combinations (all true or all false) -/
def invalid_true_false_combinations : ℕ := 2

/-- The main theorem stating the number of ways to create the answer key -/
theorem answer_key_combinations : 
  (total_true_false_combinations - invalid_true_false_combinations) * 
  (multiple_choice_options^multiple_choice_questions) = 96 := by
  sorry

end answer_key_combinations_l1866_186678


namespace june_initial_stickers_l1866_186656

/-- The number of stickers June had initially -/
def june_initial : ℕ := 76

/-- The number of stickers Bonnie had initially -/
def bonnie_initial : ℕ := 63

/-- The number of stickers their grandparents gave to each of them -/
def gift : ℕ := 25

/-- The combined total of stickers after receiving the gifts -/
def total : ℕ := 189

theorem june_initial_stickers : 
  june_initial + gift + bonnie_initial + gift = total := by sorry

end june_initial_stickers_l1866_186656


namespace interest_rate_calculation_l1866_186616

/-- Given a principal amount that yields 202.50 interest at 4.5% rate, 
    prove that the rate yielding 225 interest on the same principal is 5% -/
theorem interest_rate_calculation (P : ℝ) : 
  P * 0.045 = 202.50 → P * (5 / 100) = 225 := by
  sorry

end interest_rate_calculation_l1866_186616


namespace sabrina_cookies_left_l1866_186646

/-- Calculates the number of cookies Sabrina has left after a series of transactions -/
def cookies_left (initial : ℕ) (to_brother : ℕ) (fathers_cookies : ℕ) : ℕ :=
  let after_brother := initial - to_brother
  let from_mother := 3 * to_brother
  let after_mother := after_brother + from_mother
  let to_sister := after_mother / 3
  let after_sister := after_mother - to_sister
  let from_father := fathers_cookies / 4
  let after_father := after_sister + from_father
  let to_cousin := after_father / 2
  after_father - to_cousin

/-- Theorem stating that Sabrina is left with 18 cookies -/
theorem sabrina_cookies_left :
  cookies_left 28 10 16 = 18 := by
  sorry

end sabrina_cookies_left_l1866_186646


namespace train_length_calculation_l1866_186687

/-- Calculates the length of a train given the speeds of two trains, time to cross, and length of the other train -/
theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (l2 : ℝ) (h1 : v1 = 120) (h2 : v2 = 80) (h3 : t = 9) (h4 : l2 = 410.04) :
  let relative_speed := (v1 + v2) * 1000 / 3600
  let total_length := relative_speed * t
  let l1 := total_length - l2
  l1 = 90 := by sorry

end train_length_calculation_l1866_186687


namespace total_legs_is_22_l1866_186693

/-- The number of legs for each type of animal -/
def dog_legs : ℕ := 4
def bird_legs : ℕ := 2
def insect_legs : ℕ := 6

/-- The number of each type of animal -/
def num_dogs : ℕ := 3
def num_birds : ℕ := 2
def num_insects : ℕ := 2

/-- The total number of legs -/
def total_legs : ℕ := num_dogs * dog_legs + num_birds * bird_legs + num_insects * insect_legs

theorem total_legs_is_22 : total_legs = 22 := by
  sorry

end total_legs_is_22_l1866_186693


namespace value_of_y_l1866_186698

theorem value_of_y : ∃ y : ℝ, (3 * y) / 4 = 15 ∧ y = 20 := by
  sorry

end value_of_y_l1866_186698


namespace imaginary_part_of_one_minus_i_squared_l1866_186614

theorem imaginary_part_of_one_minus_i_squared : Complex.im ((1 - Complex.I) ^ 2) = -2 := by sorry

end imaginary_part_of_one_minus_i_squared_l1866_186614


namespace range_of_m_l1866_186672

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q,
    prove that the range of values for m is m > 2/3. -/
theorem range_of_m (p q : ℝ → Prop) (m : ℝ) 
  (h_p : ∀ x, p x ↔ (x + 1) * (x - 1) ≤ 0)
  (h_q : ∀ x, q x ↔ (x + 1) * (x - (3 * m - 1)) ≤ 0)
  (h_m_pos : m > 0)
  (h_sufficient : ∀ x, p x → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬p x) :
  m > 2/3 := by
  sorry

end range_of_m_l1866_186672


namespace total_cost_is_985_l1866_186602

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℚ := 3.75

/-- The additional cost of a train ride compared to a bus ride -/
def train_extra_cost : ℚ := 2.35

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℚ := bus_cost + (bus_cost + train_extra_cost)

/-- Theorem stating that the total cost of one train ride and one bus ride is $9.85 -/
theorem total_cost_is_985 : total_cost = 9.85 := by
  sorry

end total_cost_is_985_l1866_186602


namespace ellipse_equation_l1866_186631

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  c : ℝ  -- Distance from center to focus
  h : c < a  -- Ensure c is less than a

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / (e.a^2 - e.c^2) = 1

/-- Theorem: Given the conditions, prove the standard equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.a + e.c = 3)  -- Distance to one focus is 3
  (h2 : e.a - e.c = 1)  -- Distance to the other focus is 1
  (x y : ℝ) :
  standard_equation e x y ↔ x^2/4 + y^2/3 = 1 := by
  sorry

end ellipse_equation_l1866_186631


namespace battery_change_month_l1866_186629

/-- Given a 7-month interval between battery changes, starting in January,
    prove that the 15th change will occur in March. -/
theorem battery_change_month :
  let interval := 7  -- months between changes
  let start_month := 1  -- January
  let change_number := 15
  let total_months := interval * (change_number - 1)
  let years_passed := total_months / 12
  let extra_months := total_months % 12
  (start_month + extra_months - 1) % 12 + 1 = 3  -- 3 represents March
  := by sorry

end battery_change_month_l1866_186629


namespace reciprocal_of_hcf_24_182_l1866_186641

theorem reciprocal_of_hcf_24_182 : 
  let a : ℕ := 24
  let b : ℕ := 182
  let hcf := Nat.gcd a b
  1 / (hcf : ℚ) = 1 / 2 := by sorry

end reciprocal_of_hcf_24_182_l1866_186641


namespace same_solution_implies_m_half_l1866_186667

theorem same_solution_implies_m_half 
  (h1 : ∃ x, 4*x + 2*m = 3*x + 1)
  (h2 : ∃ x, 3*x + 2*m = 6*x + 1)
  (h3 : ∃ x, (4*x + 2*m = 3*x + 1) ∧ (3*x + 2*m = 6*x + 1)) :
  m = 1/2 := by
sorry

end same_solution_implies_m_half_l1866_186667


namespace sum_of_decimals_l1866_186644

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 := by
  sorry

end sum_of_decimals_l1866_186644


namespace circle_equation_radius_l1866_186680

theorem circle_equation_radius (x y d : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 2*y + d = 0) → 
  (∃ h k : ℝ, ∀ x y, (x - h)^2 + (y - k)^2 = 5^2) →
  d = -8 := by
sorry

end circle_equation_radius_l1866_186680


namespace fence_cost_per_foot_l1866_186684

/-- Proves that for a square plot with an area of 289 sq ft and a total fencing cost of Rs. 3876, the price per foot of fencing is Rs. 57. -/
theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) (h1 : area = 289) (h2 : total_cost = 3876) :
  (total_cost / (4 * Real.sqrt area)) = 57 := by
sorry

end fence_cost_per_foot_l1866_186684


namespace parallel_line_length_l1866_186691

theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 18 → 
  (parallel_line / base)^2 = 1/2 → 
  parallel_line = 9 * Real.sqrt 2 :=
by sorry

end parallel_line_length_l1866_186691


namespace correct_calculation_l1866_186679

theorem correct_calculation : (36 - 12) / (3 / 2) = 16 := by
  sorry

end correct_calculation_l1866_186679


namespace odd_function_iff_graph_symmetry_solution_exists_when_p_zero_more_than_two_solutions_l1866_186601

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Statement 1: f(x) is an odd function if and only if q = 0
theorem odd_function_iff (p q : ℝ) :
  (∀ x : ℝ, f p q (-x) = -(f p q x)) ↔ q = 0 := by sorry

-- Statement 2: The graph of f(x) is symmetric about the point (0, q)
theorem graph_symmetry (p q : ℝ) :
  ∀ x : ℝ, f p q (x) - q = -(f p q (-x) - q) := by sorry

-- Statement 3: When p = 0, the equation f(x) = 0 always has at least one solution
theorem solution_exists_when_p_zero (q : ℝ) :
  ∃ x : ℝ, f 0 q x = 0 := by sorry

-- Statement 4: There exists a combination of p and q such that f(x) = 0 has more than two solutions
theorem more_than_two_solutions :
  ∃ p q : ℝ, ∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (f p q x₁ = 0 ∧ f p q x₂ = 0 ∧ f p q x₃ = 0) := by sorry

end odd_function_iff_graph_symmetry_solution_exists_when_p_zero_more_than_two_solutions_l1866_186601


namespace absolute_value_equation_solution_l1866_186628

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ y => |y - 8| + 3 * y
  ∃ (y₁ y₂ : ℝ), y₁ = 23/4 ∧ y₂ = 7/2 ∧ f y₁ = 15 ∧ f y₂ = 15 ∧
    (∀ y : ℝ, f y = 15 → y = y₁ ∨ y = y₂) :=
by sorry

end absolute_value_equation_solution_l1866_186628


namespace bouquet_cost_proportional_cost_of_25_lilies_l1866_186603

/-- The cost of a bouquet of lilies -/
def bouquet_cost (lilies : ℕ) : ℝ :=
  sorry

/-- The number of lilies in the first bouquet -/
def lilies₁ : ℕ := 15

/-- The cost of the first bouquet -/
def cost₁ : ℝ := 30

/-- The number of lilies in the second bouquet -/
def lilies₂ : ℕ := 25

theorem bouquet_cost_proportional :
  ∀ (n m : ℕ), n ≠ 0 → m ≠ 0 →
  bouquet_cost n / n = bouquet_cost m / m :=
  sorry

theorem cost_of_25_lilies :
  bouquet_cost lilies₂ = 50 :=
  sorry

end bouquet_cost_proportional_cost_of_25_lilies_l1866_186603


namespace smallest_valid_number_l1866_186669

def is_valid_number (x : ℕ) : Prop :=
  x > 0 ∧ 
  ∃ (multiples : Finset ℕ), 
    multiples.card = 10 ∧ 
    ∀ m ∈ multiples, 
      m < 100 ∧ 
      m % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ m = k * x

theorem smallest_valid_number : 
  ∀ y : ℕ, y < 3 → ¬(is_valid_number y) ∧ is_valid_number 3 :=
sorry

end smallest_valid_number_l1866_186669


namespace largest_n_for_factorization_l1866_186633

/-- 
Theorem: The largest value of n for which 6x^2 + nx + 72 can be factored 
as (6x + A)(x + B), where A and B are integers, is 433.
-/
theorem largest_n_for_factorization : 
  (∃ n : ℤ, ∀ m : ℤ, 
    (∃ A B : ℤ, ∀ x : ℚ, 6 * x^2 + n * x + 72 = (6 * x + A) * (x + B)) ∧
    (∃ A B : ℤ, ∀ x : ℚ, 6 * x^2 + m * x + 72 = (6 * x + A) * (x + B)) →
    m ≤ n) ∧
  (∃ A B : ℤ, ∀ x : ℚ, 6 * x^2 + 433 * x + 72 = (6 * x + A) * (x + B)) :=
by sorry

end largest_n_for_factorization_l1866_186633


namespace compound_oxygen_atoms_l1866_186604

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  al : ℕ
  p : ℕ
  o : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (alWeight pWeight oWeight : ℕ) : ℕ :=
  c.al * alWeight + c.p * pWeight + c.o * oWeight

/-- Theorem stating the relationship between the compound composition and its molecular weight -/
theorem compound_oxygen_atoms (alWeight pWeight oWeight : ℕ) (c : Compound) :
  alWeight = 27 ∧ pWeight = 31 ∧ oWeight = 16 ∧ c.al = 1 ∧ c.p = 1 →
  (molecularWeight c alWeight pWeight oWeight = 122 ↔ c.o = 4) :=
by sorry

end compound_oxygen_atoms_l1866_186604


namespace composition_ratio_l1866_186652

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 3 * x - 2

theorem composition_ratio : (f (g (f 2))) / (g (f (g 2))) = 41 / 31 := by
  sorry

end composition_ratio_l1866_186652


namespace gcf_of_90_and_135_l1866_186692

theorem gcf_of_90_and_135 : Nat.gcd 90 135 = 45 := by
  sorry

end gcf_of_90_and_135_l1866_186692


namespace triangle_translation_l1866_186621

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a translation vector
structure Translation :=
  (dx : ℝ)
  (dy : ℝ)

-- Define a function to apply a translation to a point
def translate (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

-- The main theorem
theorem triangle_translation
  (A B C A' : Point)
  (h_A : A = { x := -1, y := -4 })
  (h_B : B = { x := 1, y := 1 })
  (h_C : C = { x := -1, y := 4 })
  (h_A' : A' = { x := 1, y := -1 })
  (h_translation : ∃ t : Translation, translate A t = A') :
  ∃ (B' C' : Point),
    B' = { x := 3, y := 4 } ∧
    C' = { x := 1, y := 7 } ∧
    translate B h_translation.choose = B' ∧
    translate C h_translation.choose = C' :=
sorry

end triangle_translation_l1866_186621


namespace quadratic_function_conditions_l1866_186606

/-- A quadratic function passing through (1,-4) with vertex at (-1,0) -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 1

/-- Theorem stating that f satisfies the given conditions -/
theorem quadratic_function_conditions :
  (f 1 = -4) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, f x = a * (x + 1)^2) := by
  sorry

#check quadratic_function_conditions

end quadratic_function_conditions_l1866_186606


namespace question_1_l1866_186697

theorem question_1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) :
  a^2 + 3/2 * b - 5 = -2 := by sorry

end question_1_l1866_186697


namespace letter_150_is_B_l1866_186685

def letter_sequence : ℕ → Char
  | n => match n % 4 with
    | 0 => 'A'
    | 1 => 'B'
    | 2 => 'C'
    | _ => 'D'

theorem letter_150_is_B : letter_sequence 149 = 'B' := by
  sorry

end letter_150_is_B_l1866_186685


namespace evaluate_expression_l1866_186607

theorem evaluate_expression : (-1 : ℤ) ^ (3^3) + 1 ^ (3^3) = 0 := by sorry

end evaluate_expression_l1866_186607


namespace watch_cost_price_l1866_186660

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (C : ℚ),
  (C * 88 / 100 : ℚ) + 140 = C * 104 / 100 ∧ C = 875 := by
  sorry

end watch_cost_price_l1866_186660


namespace count_numbers_greater_than_three_l1866_186674

theorem count_numbers_greater_than_three : 
  let numbers : Finset ℝ := {0.8, 1/2, 0.9, 1/3}
  (numbers.filter (λ x => x > 3)).card = 0 := by
sorry

end count_numbers_greater_than_three_l1866_186674


namespace largest_prime_factor_of_sum_of_factorials_l1866_186673

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 5 + factorial 6 + factorial 7

theorem largest_prime_factor_of_sum_of_factorials :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_factorials ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_factorials → q ≤ p :=
by sorry

end largest_prime_factor_of_sum_of_factorials_l1866_186673


namespace no_solution_exists_l1866_186609

theorem no_solution_exists (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * (y + z) + y * (z + x) = y * (z + x) + z * (x + y) → 
  False :=
sorry

end no_solution_exists_l1866_186609


namespace solution_set_of_even_increasing_function_l1866_186676

noncomputable def f (a b x : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_of_even_increasing_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_incr : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  ∀ x, f a b (2 - x) > 0 ↔ x < 0 ∨ x > 4 :=
sorry

end solution_set_of_even_increasing_function_l1866_186676


namespace polynomial_value_impossibility_l1866_186665

theorem polynomial_value_impossibility
  (P : ℤ → ℤ)  -- P is a function from integers to integers
  (h_poly : ∃ (Q : ℤ → ℤ), ∀ x, P x = Q x)  -- P is a polynomial
  (a b c d : ℤ)  -- a, b, c, d are integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)  -- a, b, c, d are distinct
  (h_values : P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5)  -- P(a) = P(b) = P(c) = P(d) = 5
  : ¬ ∃ (k : ℤ), P k = 8 :=  -- There is no integer k such that P(k) = 8
by sorry

end polynomial_value_impossibility_l1866_186665


namespace construction_contract_l1866_186623

theorem construction_contract (H : ℕ) 
  (first_half : H * 3 / 5 = H - (300 + 500))
  (remaining : 500 = H - (H * 3 / 5 + 300)) : H = 2000 := by
  sorry

end construction_contract_l1866_186623


namespace parabola_unique_coefficients_l1866_186699

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the parabola at a given x -/
def Parabola.eval (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Calculates the derivative of the parabola at a given x -/
def Parabola.derivative (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem parabola_unique_coefficients :
  ∀ p : Parabola,
    p.eval 1 = 1 →                        -- Parabola passes through (1, 1)
    p.eval 2 = -1 →                       -- Parabola passes through (2, -1)
    p.derivative 2 = 1 →                  -- Tangent line at (2, -1) has slope 1
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end parabola_unique_coefficients_l1866_186699


namespace standard_pairs_parity_l1866_186659

/-- Represents the color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard with m rows and n columns -/
structure Chessboard (m n : ℕ) where
  colors : Fin m → Fin n → Color
  m_ge_3 : m ≥ 3
  n_ge_3 : n ≥ 3

/-- Count of blue squares on the edges (excluding corners) of the chessboard -/
def count_edge_blue (board : Chessboard m n) : ℕ := sorry

/-- Count of standard pairs (adjacent squares with different colors) on the chessboard -/
def count_standard_pairs (board : Chessboard m n) : ℕ := sorry

/-- Main theorem: The number of standard pairs is odd iff the number of blue edge squares is odd -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Odd (count_standard_pairs board) ↔ Odd (count_edge_blue board) := by sorry

end standard_pairs_parity_l1866_186659


namespace chicken_price_per_pound_l1866_186695

-- Define the given values
def num_steaks : ℕ := 4
def steak_weight : ℚ := 1/2
def steak_price_per_pound : ℚ := 15
def chicken_weight : ℚ := 3/2
def total_spent : ℚ := 42

-- Define the theorem
theorem chicken_price_per_pound :
  (total_spent - (num_steaks * steak_weight * steak_price_per_pound)) / chicken_weight = 8 := by
  sorry

end chicken_price_per_pound_l1866_186695


namespace no_two_unique_digit_cubes_l1866_186661

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

def no_common_digits (n m : ℕ) : Prop :=
  (n.digits 10).toFinset ∩ (m.digits 10).toFinset = ∅

theorem no_two_unique_digit_cubes (kub : ℕ) 
  (h1 : is_three_digit_number kub)
  (h2 : has_unique_digits kub)
  (h3 : is_cube kub) :
  ¬ ∃ shar : ℕ, 
    is_three_digit_number shar ∧ 
    has_unique_digits shar ∧ 
    is_cube shar ∧ 
    no_common_digits kub shar :=
by sorry

end no_two_unique_digit_cubes_l1866_186661


namespace x_fifth_minus_ten_x_l1866_186608

theorem x_fifth_minus_ten_x (x : ℝ) : x = 5 → x^5 - 10*x = 3075 := by
  sorry

end x_fifth_minus_ten_x_l1866_186608


namespace necessary_but_not_sufficient_l1866_186634

theorem necessary_but_not_sufficient (x y : ℝ) :
  (¬ ((x > 3) ∨ (y > 2)) → ¬ (x + y > 5)) ∧
  ¬ ((x > 3) ∨ (y > 2) → (x + y > 5)) := by
  sorry

end necessary_but_not_sufficient_l1866_186634


namespace elise_cab_ride_cost_l1866_186654

/-- Calculates the total cost of a cab ride given the base price, cost per mile, and distance traveled. -/
def cab_ride_cost (base_price : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_price + cost_per_mile * distance

/-- Proves that Elise's cab ride cost $23 -/
theorem elise_cab_ride_cost : cab_ride_cost 3 4 5 = 23 := by
  sorry

end elise_cab_ride_cost_l1866_186654


namespace inequality_solution_l1866_186686

theorem inequality_solution : 
  let x : ℝ := 3
  (1/3 - x/3 : ℝ) < -1/2 :=
by sorry

end inequality_solution_l1866_186686


namespace imaginary_part_of_complex_division_l1866_186670

theorem imaginary_part_of_complex_division (z : ℂ) : 
  z = (3 + 4 * I) / I → Complex.im z = -3 := by
  sorry

end imaginary_part_of_complex_division_l1866_186670


namespace square_difference_given_sum_and_product_l1866_186696

theorem square_difference_given_sum_and_product (m n : ℝ) 
  (h1 : m + n = 6) (h2 : m * n = 4) : (m - n)^2 = 20 := by
  sorry

end square_difference_given_sum_and_product_l1866_186696


namespace max_value_theorem_l1866_186649

theorem max_value_theorem (t x1 x2 : ℝ) : 
  t > 2 → x2 > x1 → x1 > 0 → 
  (Real.exp x1 - x1 = t) → (x2 - Real.log x2 = t) →
  (∃ (c : ℝ), c = Real.log t / (x2 - x1) ∧ c ≤ 1 / Real.exp 1 ∧ 
   ∀ (y1 y2 : ℝ), y2 > y1 → y1 > 0 → 
   (Real.exp y1 - y1 = t) → (y2 - Real.log y2 = t) →
   Real.log t / (y2 - y1) ≤ c) :=
by sorry

end max_value_theorem_l1866_186649


namespace unique_valid_sequence_l1866_186651

/-- Represents a sequence of positive integers satisfying the given conditions -/
def ValidSequence (a : Fin 5 → ℕ+) : Prop :=
  a 0 = 1 ∧
  (99 : ℚ) / 100 = (a 0 : ℚ) / a 1 + (a 1 : ℚ) / a 2 + (a 2 : ℚ) / a 3 + (a 3 : ℚ) / a 4 ∧
  ∀ k : Fin 3, ((a (k + 1) : ℕ) - 1) * (a (k - 1) : ℕ) ≥ (a k : ℕ)^2 * ((a k : ℕ) - 1)

/-- The theorem stating that there is only one valid sequence -/
theorem unique_valid_sequence :
  ∃! a : Fin 5 → ℕ+, ValidSequence a ∧
    a 0 = 1 ∧ a 1 = 2 ∧ a 2 = 5 ∧ a 3 = 56 ∧ a 4 = 25^2 * 56 := by
  sorry

end unique_valid_sequence_l1866_186651


namespace coin_distribution_formula_l1866_186600

/-- An arithmetic sequence representing the distribution of coins among people. -/
def CoinDistribution (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem coin_distribution_formula 
  (a₁ d : ℚ) 
  (h1 : (CoinDistribution a₁ d 1) + (CoinDistribution a₁ d 2) = 
        (CoinDistribution a₁ d 3) + (CoinDistribution a₁ d 4) + (CoinDistribution a₁ d 5))
  (h2 : (CoinDistribution a₁ d 1) + (CoinDistribution a₁ d 2) + (CoinDistribution a₁ d 3) + 
        (CoinDistribution a₁ d 4) + (CoinDistribution a₁ d 5) = 5) :
  ∀ n : ℕ, n ≥ 1 → n ≤ 5 → CoinDistribution a₁ d n = -1/6 * n + 3/2 :=
sorry

end coin_distribution_formula_l1866_186600


namespace complex_number_location_l1866_186650

theorem complex_number_location (z : ℂ) (h : (3 - 2*I)*z = 4 + 3*I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end complex_number_location_l1866_186650


namespace largest_n_value_l1866_186612

/-- Represents a digit in base 8 or 9 -/
def Digit := Fin 9

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (a b c : Digit) : ℕ :=
  64 * a.val + 8 * b.val + c.val

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (c b a : Digit) : ℕ :=
  81 * c.val + 9 * b.val + a.val

/-- Checks if a number is even -/
def isEven (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

theorem largest_n_value (a b c : Digit) 
    (h1 : base8ToBase10 a b c = base9ToBase10 c b a)
    (h2 : isEven c.val)
    (h3 : ∀ x y z : Digit, 
      base8ToBase10 x y z = base9ToBase10 z y x → 
      isEven z.val → 
      base8ToBase10 x y z ≤ base8ToBase10 a b c) :
  base8ToBase10 a b c = 120 := by
  sorry

end largest_n_value_l1866_186612


namespace central_angle_of_specific_sector_l1866_186662

/-- A circular sector with given circumference and area -/
structure CircularSector where
  circumference : ℝ
  area : ℝ

/-- The possible central angles of a circular sector -/
def central_angles (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + r * θ = s.circumference ∧ 1/2 * r^2 * θ = s.area}

/-- Theorem: The central angle of a sector with circumference 6 and area 2 is either 1 or 4 -/
theorem central_angle_of_specific_sector :
  let s : CircularSector := ⟨6, 2⟩
  central_angles s = {1, 4} := by sorry

end central_angle_of_specific_sector_l1866_186662


namespace g_prime_zero_f_symmetry_f_prime_symmetry_l1866_186620

-- Define the functions and their derivatives
variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)

-- Define the conditions
axiom func_relation : ∀ x, f (x + 3) = g (-x) + 4
axiom deriv_relation : ∀ x, f' x + g' (1 + x) = 0
axiom g_even : ∀ x, g (2*x + 1) = g (-2*x + 1)

-- Define the derivative relationship
axiom f_deriv : ∀ x, (deriv f) x = f' x
axiom g_deriv : ∀ x, (deriv g) x = g' x

-- State the theorems to be proved
theorem g_prime_zero : g' 1 = 0 := by sorry

theorem f_symmetry : ∀ x, f (x + 4) = f x := by sorry

theorem f_prime_symmetry : ∀ x, f' (x + 2) = f' x := by sorry

end g_prime_zero_f_symmetry_f_prime_symmetry_l1866_186620


namespace eight_reader_permutations_l1866_186639

theorem eight_reader_permutations : Nat.factorial 8 = 40320 := by
  sorry

end eight_reader_permutations_l1866_186639


namespace ball_drawing_exclusivity_l1866_186625

structure Ball :=
  (color : String)

def Bag := Multiset Ball

def draw (bag : Bag) (n : ℕ) := Multiset Ball

def atLeastOneWhite (draw : Multiset Ball) : Prop := sorry
def bothWhite (draw : Multiset Ball) : Prop := sorry
def atLeastOneRed (draw : Multiset Ball) : Prop := sorry
def exactlyOneWhite (draw : Multiset Ball) : Prop := sorry
def exactlyTwoWhite (draw : Multiset Ball) : Prop := sorry
def bothRed (draw : Multiset Ball) : Prop := sorry

def mutuallyExclusive (e1 e2 : Multiset Ball → Prop) : Prop := sorry

def initialBag : Bag := sorry

theorem ball_drawing_exclusivity :
  let result := draw initialBag 2
  (mutuallyExclusive (exactlyOneWhite) (exactlyTwoWhite)) ∧
  (mutuallyExclusive (atLeastOneWhite) (bothRed)) ∧
  ¬(mutuallyExclusive (atLeastOneWhite) (bothWhite)) ∧
  ¬(mutuallyExclusive (atLeastOneWhite) (atLeastOneRed)) := by sorry

end ball_drawing_exclusivity_l1866_186625


namespace fish_feeding_cost_l1866_186647

/-- Calculates the total cost to feed fish for 30 days given the specified conditions --/
theorem fish_feeding_cost :
  let goldfish_count : ℕ := 50
  let koi_count : ℕ := 30
  let guppies_count : ℕ := 20
  let goldfish_food : ℚ := 1.5
  let koi_food : ℚ := 2.5
  let guppies_food : ℚ := 0.75
  let goldfish_special_food_ratio : ℚ := 0.25
  let koi_special_food_ratio : ℚ := 0.4
  let guppies_special_food_ratio : ℚ := 0.1
  let special_food_cost_goldfish : ℚ := 3
  let special_food_cost_others : ℚ := 4
  let regular_food_cost : ℚ := 2
  let days : ℕ := 30

  (goldfish_count * goldfish_food * (goldfish_special_food_ratio * special_food_cost_goldfish +
    (1 - goldfish_special_food_ratio) * regular_food_cost) +
   koi_count * koi_food * (koi_special_food_ratio * special_food_cost_others +
    (1 - koi_special_food_ratio) * regular_food_cost) +
   guppies_count * guppies_food * (guppies_special_food_ratio * special_food_cost_others +
    (1 - guppies_special_food_ratio) * regular_food_cost)) * days = 12375 :=
by sorry


end fish_feeding_cost_l1866_186647


namespace number_exists_l1866_186690

theorem number_exists : ∃ x : ℝ, 0.6667 * x - 10 = 0.25 * x := by
  sorry

end number_exists_l1866_186690


namespace sqrt_fraction_simplification_l1866_186637

theorem sqrt_fraction_simplification :
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = 17 * Real.sqrt 41 / 41 := by
  sorry

end sqrt_fraction_simplification_l1866_186637


namespace wang_house_number_l1866_186635

def is_valid_triplet (a b c : ℕ) : Prop :=
  a * b * c = 40 ∧ a > 0 ∧ b > 0 ∧ c > 0

def house_number (a b c : ℕ) : ℕ := a + b + c

def is_ambiguous (n : ℕ) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂, 
    is_valid_triplet a₁ b₁ c₁ ∧ 
    is_valid_triplet a₂ b₂ c₂ ∧ 
    house_number a₁ b₁ c₁ = n ∧ 
    house_number a₂ b₂ c₂ = n ∧ 
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)

theorem wang_house_number : 
  ∃! n, is_ambiguous n ∧ ∀ m, is_ambiguous m → m = n :=
by
  sorry

end wang_house_number_l1866_186635


namespace roberto_outfits_l1866_186645

/-- The number of different outfits Roberto can put together -/
def number_of_outfits (trousers shirts jackets : ℕ) (incompatible_combinations : ℕ) : ℕ :=
  trousers * shirts * jackets - incompatible_combinations * shirts

/-- Theorem stating the number of outfits Roberto can put together -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 7
  let jackets : ℕ := 4
  let incompatible_combinations : ℕ := 1
  number_of_outfits trousers shirts jackets incompatible_combinations = 133 := by
sorry

end roberto_outfits_l1866_186645


namespace fruit_basket_count_l1866_186627

def num_apples : ℕ := 7
def num_oranges : ℕ := 12
def min_fruits_per_basket : ℕ := 2

-- Function to calculate the number of valid fruit baskets
def count_valid_baskets (apples oranges min_fruits : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - (1 + apples + oranges)

-- Theorem stating that the number of valid fruit baskets is 101
theorem fruit_basket_count :
  count_valid_baskets num_apples num_oranges min_fruits_per_basket = 101 := by
  sorry

#eval count_valid_baskets num_apples num_oranges min_fruits_per_basket

end fruit_basket_count_l1866_186627


namespace sum_of_ages_seven_years_hence_l1866_186688

-- Define X's current age
def X_current : ℕ := 45

-- Define Y's current age as a function of X's current age
def Y_current : ℕ := X_current - 21

-- Theorem to prove
theorem sum_of_ages_seven_years_hence : 
  X_current + Y_current + 14 = 83 :=
by sorry

end sum_of_ages_seven_years_hence_l1866_186688


namespace common_tangents_count_l1866_186653

/-- The number of common tangents between two circles -/
def num_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ := sorry

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- Theorem stating that the number of common tangents between C₁ and C₂ is 4 -/
theorem common_tangents_count : num_common_tangents C₁ C₂ = 4 := by sorry

end common_tangents_count_l1866_186653


namespace parallel_vectors_imply_right_triangle_l1866_186605

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if vectors m = (a+c, b) and n = (b, a-c) are parallel, then ABC is a right triangle -/
theorem parallel_vectors_imply_right_triangle 
  (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_parallel : (a + c) * (a - c) = b^2) :
  a^2 = b^2 + c^2 := by
sorry

end parallel_vectors_imply_right_triangle_l1866_186605


namespace talent_show_participants_l1866_186689

theorem talent_show_participants (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 34 →
  difference = 22 →
  girls = (total + difference) / 2 →
  girls = 28 :=
by
  sorry

end talent_show_participants_l1866_186689


namespace fraction_equality_l1866_186681

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (1/x - 1/y) / (1/x + 1/y) = 1001 → (x + y) / (x - y) = -(1/1001) := by
  sorry

end fraction_equality_l1866_186681


namespace max_value_expression_l1866_186632

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * z * Real.sqrt 3 + 9 * y * z ≤ Real.sqrt ((29 * 54) / 5) ∧
  ∃ (x_max y_max z_max : ℝ),
    x_max ≥ 0 ∧ y_max ≥ 0 ∧ z_max ≥ 0 ∧
    x_max^2 + y_max^2 + z_max^2 = 1 ∧
    3 * x_max * z_max * Real.sqrt 3 + 9 * y_max * z_max = Real.sqrt ((29 * 54) / 5) :=
by sorry

end max_value_expression_l1866_186632


namespace female_democrats_count_l1866_186618

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 810 →
  female + male = total →
  (female / 2 + male / 4 : ℚ) = total / 3 →
  female / 2 = 135 :=
by sorry

end female_democrats_count_l1866_186618


namespace highlight_film_average_time_l1866_186663

def point_guard_time : ℕ := 130
def shooting_guard_time : ℕ := 145
def small_forward_time : ℕ := 85
def power_forward_time : ℕ := 60
def center_time : ℕ := 180
def number_of_players : ℕ := 5

def total_time : ℕ := point_guard_time + shooting_guard_time + small_forward_time + power_forward_time + center_time

theorem highlight_film_average_time :
  (total_time / number_of_players : ℚ) / 60 = 2 := by
  sorry

end highlight_film_average_time_l1866_186663


namespace line_through_points_l1866_186611

/-- Theorem: For a line y = ax + b passing through points (3, 4) and (10, 22), a - b = 6 2/7 -/
theorem line_through_points (a b : ℚ) : 
  (4 : ℚ) = a * 3 + b ∧ (22 : ℚ) = a * 10 + b → a - b = (44 : ℚ) / 7 := by
  sorry

end line_through_points_l1866_186611


namespace math_club_attendance_l1866_186610

theorem math_club_attendance (total_sessions : Nat) (students_per_session : Nat)
  (three_session_attendees : Nat) (two_session_attendees : Nat) (one_session_attendees : Nat)
  (h1 : total_sessions = 4)
  (h2 : students_per_session = 20)
  (h3 : three_session_attendees = 9)
  (h4 : two_session_attendees = 5)
  (h5 : one_session_attendees = 3) :
  ∃ (all_session_attendees : Nat),
    all_session_attendees * total_sessions +
    three_session_attendees * 3 +
    two_session_attendees * 2 +
    one_session_attendees * 1 =
    total_sessions * students_per_session ∧
    all_session_attendees = 10 := by
  sorry

end math_club_attendance_l1866_186610


namespace smallest_n_for_Q_less_than_1_over_3020_l1866_186613

def Q (n : ℕ+) : ℚ := (Nat.factorial (3*n-1)) / (Nat.factorial (3*n+1))

theorem smallest_n_for_Q_less_than_1_over_3020 :
  ∀ k : ℕ+, k < 19 → Q k ≥ 1/3020 ∧ Q 19 < 1/3020 := by sorry

end smallest_n_for_Q_less_than_1_over_3020_l1866_186613


namespace rice_on_eighth_day_l1866_186636

/-- Represents the number of laborers on a given day -/
def laborers (day : ℕ) : ℕ := 64 + 7 * (day - 1)

/-- The amount of rice given to each laborer per day -/
def ricePerLaborer : ℕ := 3

/-- The amount of rice given out on a specific day -/
def riceOnDay (day : ℕ) : ℕ := laborers day * ricePerLaborer

theorem rice_on_eighth_day : riceOnDay 8 = 339 := by
  sorry

end rice_on_eighth_day_l1866_186636


namespace fraction_power_multiplication_l1866_186617

theorem fraction_power_multiplication :
  (3 / 5 : ℝ)^4 * (2 / 9 : ℝ)^(1/2) = 81 * Real.sqrt 2 / 1875 := by
  sorry

end fraction_power_multiplication_l1866_186617


namespace special_circle_properties_special_circle_unique_l1866_186615

/-- The circle passing through points A(1,-1) and B(-1,1) with its center on the line x+y-2=0 -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

theorem special_circle_properties :
  ∀ x y : ℝ,
    special_circle x y →
    ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) ∧
    ∃ c_x c_y : ℝ, c_x + c_y - 2 = 0 ∧
                   (x - c_x)^2 + (y - c_y)^2 = (c_x - 1)^2 + (c_y + 1)^2 :=
by
  sorry

theorem special_circle_unique :
  ∀ f : ℝ → ℝ → Prop,
    (∀ x y : ℝ, f x y → ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1))) →
    (∀ x y : ℝ, f x y → ∃ c_x c_y : ℝ, c_x + c_y - 2 = 0 ∧
                                      (x - c_x)^2 + (y - c_y)^2 = (c_x - 1)^2 + (c_y + 1)^2) →
    ∀ x y : ℝ, f x y ↔ special_circle x y :=
by
  sorry

end special_circle_properties_special_circle_unique_l1866_186615


namespace bobby_candy_problem_l1866_186668

theorem bobby_candy_problem (initial_candy : ℕ) : 
  initial_candy + 4 + 14 = 51 → initial_candy = 33 := by
  sorry

end bobby_candy_problem_l1866_186668


namespace area_of_overlapping_squares_area_of_overlapping_squares_is_216_l1866_186677

/-- The area of the region covered by two congruent squares with side length 12 units,
    where the center of one square coincides with a vertex of the other square. -/
theorem area_of_overlapping_squares : ℝ :=
  let square_side_length : ℝ := 12
  let square_area : ℝ := square_side_length ^ 2
  let total_area : ℝ := 2 * square_area
  let overlap_area : ℝ := square_area / 2
  total_area - overlap_area

/-- The area of the region covered by two congruent squares with side length 12 units,
    where the center of one square coincides with a vertex of the other square, is 216 square units. -/
theorem area_of_overlapping_squares_is_216 : area_of_overlapping_squares = 216 := by
  sorry

end area_of_overlapping_squares_area_of_overlapping_squares_is_216_l1866_186677


namespace tech_personnel_stats_l1866_186671

def intermediate_count : ℕ := 40
def senior_count : ℕ := 10
def total_count : ℕ := intermediate_count + senior_count

def intermediate_avg : ℝ := 35
def senior_avg : ℝ := 45

def intermediate_var : ℝ := 18
def senior_var : ℝ := 73

def total_avg : ℝ := 37
def total_var : ℝ := 45

theorem tech_personnel_stats :
  (intermediate_count * intermediate_avg + senior_count * senior_avg) / total_count = total_avg ∧
  ((intermediate_count * (intermediate_var + intermediate_avg^2) + 
    senior_count * (senior_var + senior_avg^2)) / total_count - total_avg^2) = total_var :=
by sorry

end tech_personnel_stats_l1866_186671


namespace ratio_AD_DC_is_3_to_2_l1866_186682

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define point D on AC
def point_D_on_AC (A C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- Define BD = 7
def BD_equals_7 (B D : ℝ × ℝ) : Prop :=
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 7

-- Theorem statement
theorem ratio_AD_DC_is_3_to_2 
  (A B C D : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C) 
  (h_D_on_AC : point_D_on_AC A C D) 
  (h_BD : BD_equals_7 B D) : 
  ∃ (AD DC : ℝ), AD / DC = 3 / 2 := 
sorry

end ratio_AD_DC_is_3_to_2_l1866_186682


namespace modulus_of_z_l1866_186657

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define z
noncomputable def z : ℂ := 4 / (1 + i)^4 - 3 * i

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_of_z_l1866_186657


namespace apple_packing_problem_l1866_186683

theorem apple_packing_problem (apples_per_crate : ℕ) (num_crates : ℕ) 
  (rotten_percentage : ℚ) (apples_per_box : ℕ) (available_boxes : ℕ) :
  apples_per_crate = 400 →
  num_crates = 35 →
  rotten_percentage = 11/100 →
  apples_per_box = 30 →
  available_boxes = 1000 →
  ∃ (boxes_needed : ℕ), 
    boxes_needed = 416 ∧ 
    boxes_needed * apples_per_box ≥ 
      (1 - rotten_percentage) * (apples_per_crate * num_crates) ∧
    (boxes_needed - 1) * apples_per_box < 
      (1 - rotten_percentage) * (apples_per_crate * num_crates) ∧
    boxes_needed ≤ available_boxes :=
by sorry

end apple_packing_problem_l1866_186683
