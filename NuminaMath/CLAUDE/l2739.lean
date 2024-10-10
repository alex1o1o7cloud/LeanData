import Mathlib

namespace bounded_region_area_l2739_273958

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 30*|x| = 360

-- Define the vertices of the parallelogram
def vertices : List (ℝ × ℝ) :=
  [(0, -30), (0, 30), (15, -30), (-15, 30)]

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ graph_equation x y}

-- Theorem statement
theorem bounded_region_area :
  MeasureTheory.volume (bounded_region) = 1800 :=
sorry

end bounded_region_area_l2739_273958


namespace expense_increase_l2739_273997

theorem expense_increase (december_salary : ℝ) (h1 : december_salary > 0) : 
  let december_mortgage := 0.4 * december_salary
  let december_expenses := december_salary - december_mortgage
  let january_salary := 1.09 * december_salary
  let january_expenses := january_salary - december_mortgage
  (january_expenses - december_expenses) / december_expenses = 0.15
  := by sorry

end expense_increase_l2739_273997


namespace circus_crowns_l2739_273966

theorem circus_crowns (feathers_per_crown : ℕ) (total_feathers : ℕ) (h1 : feathers_per_crown = 7) (h2 : total_feathers = 6538) :
  total_feathers / feathers_per_crown = 934 := by
  sorry

end circus_crowns_l2739_273966


namespace unique_divisible_number_l2739_273939

def original_number : Nat := 20172018

theorem unique_divisible_number :
  ∃! n : Nat,
    (∃ a b : Nat, a < 10 ∧ b < 10 ∧ n = a * 1000000000 + original_number * 10 + b) ∧
    n % 8 = 0 ∧
    n % 9 = 0 :=
  by sorry

end unique_divisible_number_l2739_273939


namespace root_transformation_l2739_273967

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - r₁^2 + 3*r₁ - 7 = 0) ∧ 
  (r₂^3 - r₂^2 + 3*r₂ - 7 = 0) ∧ 
  (r₃^3 - r₃^2 + 3*r₃ - 7 = 0) →
  ((3*r₁)^3 - 3*(3*r₁)^2 + 27*(3*r₁) - 189 = 0) ∧ 
  ((3*r₂)^3 - 3*(3*r₂)^2 + 27*(3*r₂) - 189 = 0) ∧ 
  ((3*r₃)^3 - 3*(3*r₃)^2 + 27*(3*r₃) - 189 = 0) :=
by sorry

end root_transformation_l2739_273967


namespace equation_solution_l2739_273957

theorem equation_solution (a c x : ℝ) : 2 * x^2 + c^2 = (a + x)^2 → x = -a + c ∨ x = -a - c := by
  sorry

end equation_solution_l2739_273957


namespace inequality_proof_l2739_273953

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := by
  sorry

end inequality_proof_l2739_273953


namespace group_size_is_16_l2739_273998

/-- The number of children whose height increases -/
def num_taller_children : ℕ := 12

/-- The height increase for each of the taller children in cm -/
def height_increase : ℕ := 8

/-- The total height increase in cm -/
def total_height_increase : ℕ := num_taller_children * height_increase

/-- The mean height increase in cm -/
def mean_height_increase : ℕ := 6

theorem group_size_is_16 :
  ∃ n : ℕ, n > 0 ∧ (total_height_increase : ℚ) / n = mean_height_increase ∧ n = 16 := by
  sorry

end group_size_is_16_l2739_273998


namespace log_inequality_solution_set_l2739_273932

-- Define the logarithm function with base 0.1
noncomputable def log_base_point_one (x : ℝ) := Real.log x / Real.log 0.1

-- State the theorem
theorem log_inequality_solution_set :
  ∀ x : ℝ, log_base_point_one (2^x - 1) < 0 ↔ x > 1 :=
by sorry

end log_inequality_solution_set_l2739_273932


namespace stream_rate_proof_l2739_273904

/-- Proves that the rate of a stream is 5 km/hr given the conditions of a boat's travel --/
theorem stream_rate_proof (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 16 →
  distance = 147 →
  time = 7 →
  (distance / time) - boat_speed = 5 := by
  sorry

#check stream_rate_proof

end stream_rate_proof_l2739_273904


namespace average_of_w_and_x_l2739_273916

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end average_of_w_and_x_l2739_273916


namespace common_ratio_is_negative_half_l2739_273921

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  not_constant : ∃ i j, a i ≠ a j
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  a_3 : a 3 = 5 / 2
  S_3 : (a 1) + (a 2) + (a 3) = 15 / 2

/-- The common ratio of the geometric sequence -/
def common_ratio (seq : GeometricSequence) : ℚ := seq.a 2 / seq.a 1

theorem common_ratio_is_negative_half (seq : GeometricSequence) : 
  common_ratio seq = -1 / 2 := by
  sorry

end common_ratio_is_negative_half_l2739_273921


namespace binary_multiplication_l2739_273977

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinary (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
    toBinary n

theorem binary_multiplication :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, true]  -- 1001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
sorry

end binary_multiplication_l2739_273977


namespace intersection_A_B_union_A_B_A_intersect_C_nonempty_l2739_273917

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 8}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 8} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 9} := by sorry

-- Theorem for the condition when A ∩ C is non-empty
theorem A_intersect_C_nonempty (a : ℝ) : (A ∩ C a).Nonempty ↔ a ≤ 8 := by sorry

end intersection_A_B_union_A_B_A_intersect_C_nonempty_l2739_273917


namespace fraction_simplification_l2739_273922

theorem fraction_simplification (x : ℝ) (h : x = 3) : 
  (x^8 + 16*x^4 + 64 + 4*x^2) / (x^4 + 8) = 89 + 36/89 := by
  sorry

end fraction_simplification_l2739_273922


namespace indeterminateNatureAndSanity_l2739_273981

-- Define the types for Transylvanians
inductive Transylvanian
| Human
| Vampire

-- Define the mental state
inductive MentalState
| Sane
| Insane

-- Define reliability
def isReliable (t : Transylvanian) (m : MentalState) : Prop :=
  (t = Transylvanian.Human ∧ m = MentalState.Sane) ∨
  (t = Transylvanian.Vampire ∧ m = MentalState.Insane)

-- Define unreliability
def isUnreliable (t : Transylvanian) (m : MentalState) : Prop :=
  (t = Transylvanian.Human ∧ m = MentalState.Insane) ∨
  (t = Transylvanian.Vampire ∧ m = MentalState.Sane)

-- Define the statement function
def statesTrue (t : Transylvanian) (m : MentalState) : Prop :=
  isReliable t m

-- Define the answer to the question "Are you reliable?"
def answersYes (t : Transylvanian) (m : MentalState) : Prop :=
  (isReliable t m ∧ statesTrue t m) ∨ (isUnreliable t m ∧ ¬statesTrue t m)

-- Theorem: It's impossible to determine the nature or sanity of a Transylvanian
-- based on their answer to the question "Are you reliable?"
theorem indeterminateNatureAndSanity (t : Transylvanian) (m : MentalState) :
  answersYes t m → 
  (∃ (t' : Transylvanian) (m' : MentalState), t' ≠ t ∨ m' ≠ m) ∧ answersYes t' m' :=
sorry


end indeterminateNatureAndSanity_l2739_273981


namespace jason_initial_cards_l2739_273943

theorem jason_initial_cards (cards_sold : ℕ) (cards_remaining : ℕ) : 
  cards_sold = 224 → cards_remaining = 452 → cards_sold + cards_remaining = 676 := by
  sorry

end jason_initial_cards_l2739_273943


namespace derivative_at_one_l2739_273969

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 2 := by
sorry

end derivative_at_one_l2739_273969


namespace tan_fec_value_l2739_273974

/-- Square ABCD with inscribed isosceles triangle AEF -/
structure SquareWithTriangle where
  /-- Side length of the square -/
  a : ℝ
  /-- Point E on side BC -/
  e : ℝ × ℝ
  /-- Point F on side CD -/
  f : ℝ × ℝ
  /-- ABCD is a square -/
  square_abcd : e.1 ≤ a ∧ e.1 ≥ 0 ∧ f.2 ≤ a ∧ f.2 ≥ 0
  /-- E is on BC -/
  e_on_bc : e.2 = 0
  /-- F is on CD -/
  f_on_cd : f.1 = a
  /-- AEF is isosceles with AE = EF -/
  isosceles_aef : (0 - e.1)^2 + e.2^2 = (a - f.1)^2 + (f.2 - 0)^2
  /-- tan(∠AEF) = 2 -/
  tan_aef : (f.2 - 0) / (f.1 - e.1) = 2

/-- The tangent of angle FEC in the described configuration is 3 - √5 -/
theorem tan_fec_value (st : SquareWithTriangle) : 
  (st.a - st.e.1) / (st.f.2 - 0) = 3 - Real.sqrt 5 := by
  sorry


end tan_fec_value_l2739_273974


namespace solution_set_of_polynomial_equation_l2739_273956

theorem solution_set_of_polynomial_equation :
  let S := {x : ℝ | x = 0 ∨ 
                   x = Real.sqrt ((5 + Real.sqrt 5) / 2) ∨ 
                   x = -Real.sqrt ((5 + Real.sqrt 5) / 2) ∨ 
                   x = Real.sqrt ((5 - Real.sqrt 5) / 2) ∨ 
                   x = -Real.sqrt ((5 - Real.sqrt 5) / 2)}
  ∀ x : ℝ, (5*x - 5*x^3 + x^5 = 0) ↔ x ∈ S :=
by sorry

end solution_set_of_polynomial_equation_l2739_273956


namespace parabolas_intersect_on_circle_l2739_273903

/-- The parabolas y = (x - 2)² and x - 5 = (y + 1)² intersect on a circle --/
theorem parabolas_intersect_on_circle :
  ∃ (r : ℝ), r^2 = 9/4 ∧
  ∀ (x y : ℝ), (y = (x - 2)^2 ∧ x - 5 = (y + 1)^2) →
    (x - 3/2)^2 + (y + 1)^2 = r^2 := by
  sorry

end parabolas_intersect_on_circle_l2739_273903


namespace largest_integer_square_4_digits_base8_l2739_273937

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def N : ℕ := 63

/-- Conversion of a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem largest_integer_square_4_digits_base8 :
  (N^2 ≥ 8^3) ∧ (N^2 < 8^4) ∧ (∀ m : ℕ, m > N → m^2 ≥ 8^4) ∧ (toBase8 N = [7, 7]) := by
  sorry

end largest_integer_square_4_digits_base8_l2739_273937


namespace carls_lawn_area_l2739_273911

/-- Represents a rectangular lawn with fence posts -/
structure FencedLawn where
  short_side : ℕ  -- Number of posts on shorter side
  long_side : ℕ   -- Number of posts on longer side
  post_spacing : ℕ -- Distance between posts in yards

/-- The total number of fence posts -/
def total_posts (lawn : FencedLawn) : ℕ :=
  2 * (lawn.short_side + lawn.long_side) - 4

/-- The area of the lawn in square yards -/
def lawn_area (lawn : FencedLawn) : ℕ :=
  (lawn.short_side - 1) * lawn.post_spacing * ((lawn.long_side - 1) * lawn.post_spacing)

/-- Theorem stating the area of Carl's lawn -/
theorem carls_lawn_area : 
  ∃ (lawn : FencedLawn), 
    lawn.short_side = 4 ∧ 
    lawn.long_side = 12 ∧ 
    lawn.post_spacing = 3 ∧ 
    total_posts lawn = 24 ∧ 
    lawn_area lawn = 243 := by
  sorry


end carls_lawn_area_l2739_273911


namespace triangle_max_side_sum_l2739_273971

theorem triangle_max_side_sum (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a = Real.sqrt 3 ∧ 
  A = 2 * π / 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  (∀ b' c' : ℝ, b' + c' ≤ b + c → b' + c' ≤ 2) :=
by sorry

end triangle_max_side_sum_l2739_273971


namespace warren_guests_calculation_l2739_273905

/-- The number of tables Warren has -/
def num_tables : ℝ := 252.0

/-- The number of guests each table can hold -/
def guests_per_table : ℝ := 4.0

/-- The total number of guests Warren can accommodate -/
def total_guests : ℝ := num_tables * guests_per_table

theorem warren_guests_calculation : total_guests = 1008 := by
  sorry

end warren_guests_calculation_l2739_273905


namespace multiple_solutions_exist_l2739_273936

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem multiple_solutions_exist :
  ∃ (c₁ c₂ : ℝ), c₁ ≠ c₂ ∧ f (f (f (f c₁))) = 2 ∧ f (f (f (f c₂))) = 2 :=
by sorry

end multiple_solutions_exist_l2739_273936


namespace playground_boys_count_l2739_273993

/-- Given a playground with children, prove that the number of boys is 44 -/
theorem playground_boys_count (total_children : ℕ) (girls : ℕ) (boys : ℕ) : 
  total_children = 97 → girls = 53 → total_children = girls + boys → boys = 44 := by
sorry

end playground_boys_count_l2739_273993


namespace fraction_equality_l2739_273995

theorem fraction_equality (a b : ℝ) (h1 : a = (2/3) * b) (h2 : b ≠ 0) : 
  (9*a + 8*b) / (6*a) = 7/2 := by
sorry

end fraction_equality_l2739_273995


namespace bus_exit_ways_10_5_l2739_273913

/-- The number of possible ways for passengers to get off a bus -/
def bus_exit_ways (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: Given 10 passengers and 5 stops, the number of possible ways
    for passengers to get off the bus is 5^10 -/
theorem bus_exit_ways_10_5 :
  bus_exit_ways 10 5 = 5^10 := by
  sorry

end bus_exit_ways_10_5_l2739_273913


namespace fixed_point_of_exponential_function_l2739_273914

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(2*x - 1) + 1
  f (1/2) = 2 := by
sorry

end fixed_point_of_exponential_function_l2739_273914


namespace prob_two_ones_twelve_dice_l2739_273982

/-- The number of dice rolled -/
def n : ℕ := 12

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice we want to show a specific result -/
def k : ℕ := 2

/-- The probability of rolling exactly k ones out of n dice -/
def prob_k_ones (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1 / sides)^k * (1 - 1 / sides)^(n - k)

theorem prob_two_ones_twelve_dice : 
  prob_k_ones n k = (66 * 5^10 : ℚ) / 6^12 := by sorry

end prob_two_ones_twelve_dice_l2739_273982


namespace first_candidate_vote_percentage_l2739_273924

/-- Proves that the first candidate received 80% of the votes in an election with two candidates -/
theorem first_candidate_vote_percentage
  (total_votes : ℕ)
  (second_candidate_votes : ℕ)
  (h_total : total_votes = 2400)
  (h_second : second_candidate_votes = 480) :
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100 = 80 := by
  sorry

end first_candidate_vote_percentage_l2739_273924


namespace last_two_digits_sum_l2739_273933

theorem last_two_digits_sum (n : ℕ) : (9^n + 11^n) % 100 = 2 :=
by
  sorry

end last_two_digits_sum_l2739_273933


namespace unshaded_area_equilateral_triangle_l2739_273975

/-- The area of the unshaded region inside an equilateral triangle, 
    whose side is the diameter of a semi-circle with radius 1. -/
theorem unshaded_area_equilateral_triangle (r : ℝ) : 
  r = 1 → 
  ∃ (A : ℝ), A = Real.sqrt 3 - π / 6 ∧ 
  A = (3 * Real.sqrt 3 / 4) * (2 * r)^2 - π * r^2 / 6 :=
by sorry

end unshaded_area_equilateral_triangle_l2739_273975


namespace trig_expression_equals_one_l2739_273963

theorem trig_expression_equals_one : 
  let cos30 := Real.cos (30 * π / 180)
  let sin60 := Real.sin (60 * π / 180)
  let sin30 := Real.sin (30 * π / 180)
  let cos60 := Real.cos (60 * π / 180)
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = 1 := by
  sorry

end trig_expression_equals_one_l2739_273963


namespace determinant_one_l2739_273990

-- Define the property that for all m and n, there exist h and k satisfying the equations
def satisfies_equations (a b c d : ℤ) : Prop :=
  ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n

-- State the theorem
theorem determinant_one (a b c d : ℤ) (h : satisfies_equations a b c d) : |a * d - b * c| = 1 := by
  sorry

end determinant_one_l2739_273990


namespace dot_product_OA_OB_is_zero_l2739_273959

theorem dot_product_OA_OB_is_zero (OA OB : ℝ × ℝ) : 
  OA = (1, -3) →
  ‖OA‖ = ‖OB‖ →
  ‖OA - OB‖ = 2 * Real.sqrt 5 →
  OA • OB = 0 := by
  sorry

end dot_product_OA_OB_is_zero_l2739_273959


namespace highest_number_on_paper_l2739_273950

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.01020408163265306 → n = 98 :=
by sorry

end highest_number_on_paper_l2739_273950


namespace binomial_coefficient_n_plus_one_choose_n_minus_one_l2739_273929

theorem binomial_coefficient_n_plus_one_choose_n_minus_one (n : ℕ+) :
  Nat.choose (n + 1) (n - 1) = n * (n + 1) / 2 := by
  sorry

end binomial_coefficient_n_plus_one_choose_n_minus_one_l2739_273929


namespace semicircle_radius_in_isosceles_triangle_l2739_273938

/-- An isosceles triangle with a semicircle inscribed -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The diameter of the semicircle is contained in the base of the triangle -/
  diameter_in_base : radius * 2 ≤ base

/-- The theorem stating the radius of the semicircle in the given isosceles triangle -/
theorem semicircle_radius_in_isosceles_triangle 
  (triangle : IsoscelesTriangleWithSemicircle) 
  (h1 : triangle.base = 20) 
  (h2 : triangle.height = 12) : 
  triangle.radius = 60 / (5 + Real.sqrt 61) :=
sorry

end semicircle_radius_in_isosceles_triangle_l2739_273938


namespace g_has_unique_zero_l2739_273940

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * (f a x + a + 1)

theorem g_has_unique_zero (a : ℝ) (h : a > 1 / Real.exp 1) :
  ∃! x, x > 0 ∧ g a x = 0 :=
sorry

end g_has_unique_zero_l2739_273940


namespace g_equals_h_intersection_at_most_one_point_l2739_273954

-- Define the functions g and h
def g (x : ℝ) : ℝ := 2 * x - 1
def h (t : ℝ) : ℝ := 2 * t - 1

-- Theorem 1: g and h are the same function
theorem g_equals_h : g = h := by sorry

-- Theorem 2: For any function f, the intersection of y = f(x) and x = 2 has at most one point
theorem intersection_at_most_one_point (f : ℝ → ℝ) :
  ∃! y, f 2 = y := by sorry

end g_equals_h_intersection_at_most_one_point_l2739_273954


namespace circle_tangent_k_range_l2739_273931

/-- The range of k for a circle with two tangents from P(2,2) -/
theorem circle_tangent_k_range :
  ∀ k : ℝ,
  (∃ x y : ℝ, x^2 + y^2 - 2*k*x - 2*y + k^2 - k = 0) →
  (∃ t₁ t₂ : ℝ × ℝ, 
    (t₁.1 - 2)^2 + (t₁.2 - 2)^2 = ((2 - k)^2 + 1) ∧
    (t₂.1 - 2)^2 + (t₂.2 - 2)^2 = ((2 - k)^2 + 1) ∧
    t₁ ≠ t₂) →
  (k ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 4) :=
by sorry

end circle_tangent_k_range_l2739_273931


namespace max_non_managers_dept_A_l2739_273985

/-- Represents a department in the company -/
inductive Department
| A
| B
| C

/-- Represents the gender of an employee -/
inductive Gender
| Male
| Female

/-- Represents the status of a manager -/
inductive ManagerStatus
| Active
| OnVacation

/-- Represents the type of non-manager employee -/
inductive NonManagerType
| FullTime
| PartTime

/-- The company structure and policies -/
structure Company where
  /-- The ratio of managers to non-managers must be greater than this for all departments -/
  baseRatio : Rat
  /-- Department A's specific ratio requirement -/
  deptARatio : Rat
  /-- Department B's specific ratio requirement -/
  deptBRatio : Rat
  /-- The minimum gender ratio (male:female) for non-managers -/
  genderRatio : Rat

/-- Represents the workforce of a department -/
structure DepartmentWorkforce where
  department : Department
  totalManagers : Nat
  activeManagers : Nat
  nonManagersMale : Nat
  nonManagersFemale : Nat
  partTimeNonManagers : Nat

/-- Main theorem to prove -/
theorem max_non_managers_dept_A (c : Company) (dA : DepartmentWorkforce) :
  c.baseRatio = 7/32 ∧
  c.deptARatio = 9/33 ∧
  c.deptBRatio = 8/34 ∧
  c.genderRatio = 1/2 ∧
  dA.department = Department.A ∧
  dA.totalManagers = 8 ∧
  dA.activeManagers = 4 →
  dA.nonManagersMale + dA.nonManagersFemale + dA.partTimeNonManagers / 2 ≤ 12 :=
sorry

end max_non_managers_dept_A_l2739_273985


namespace check_tianning_pairs_find_x_for_negative_five_evaluate_expression_for_tianning_pair_l2739_273960

-- Define Tianning pair
def is_tianning_pair (a b : ℝ) : Prop := a + b = a * b

-- Theorem 1: Checking specific pairs
theorem check_tianning_pairs :
  is_tianning_pair 3 1.5 ∧
  is_tianning_pair (-1/2) (1/3) ∧
  ¬ is_tianning_pair (3/4) 1 :=
sorry

-- Theorem 2: Finding x for (-5, x)
theorem find_x_for_negative_five :
  ∃ x, is_tianning_pair (-5) x ∧ x = 5/6 :=
sorry

-- Theorem 3: Evaluating expression for any Tianning pair
theorem evaluate_expression_for_tianning_pair (m n : ℝ) :
  is_tianning_pair m n →
  4*(m*n+m-2*(m*n-3))-2*(3*m^2-2*n)+6*m^2 = 24 :=
sorry

end check_tianning_pairs_find_x_for_negative_five_evaluate_expression_for_tianning_pair_l2739_273960


namespace combined_instruments_count_l2739_273906

/-- Represents the number of musical instruments owned by a person -/
structure Instruments where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- Calculates the total number of instruments -/
def totalInstruments (i : Instruments) : ℕ :=
  i.flutes + i.horns + i.harps

/-- Charlie's instruments -/
def charlie : Instruments :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instruments -/
def carli : Instruments :=
  { flutes := 2 * charlie.flutes,
    horns := charlie.horns / 2,
    harps := 0 }

theorem combined_instruments_count :
  totalInstruments charlie + totalInstruments carli = 7 := by
  sorry

end combined_instruments_count_l2739_273906


namespace five_heads_in_nine_flips_l2739_273999

/-- The probability of getting exactly k heads when flipping n fair coins -/
def coinFlipProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 5 heads when flipping 9 fair coins is 63/256 -/
theorem five_heads_in_nine_flips :
  coinFlipProbability 9 5 = 63 / 256 := by
  sorry

end five_heads_in_nine_flips_l2739_273999


namespace class_composition_theorem_l2739_273973

/-- Represents the number of students in a class with specific friendship conditions -/
structure ClassComposition where
  boys : ℕ
  girls : ℕ
  total_children : ℕ
  desks : ℕ

/-- Checks if the class composition satisfies the given conditions -/
def is_valid_composition (c : ClassComposition) : Prop :=
  c.boys * 2 = c.girls * 3 ∧
  c.boys + c.girls = c.total_children ∧
  c.total_children = 31 ∧
  c.desks = 19

/-- Theorem stating that the only valid class composition has 35 students -/
theorem class_composition_theorem :
  ∀ c : ClassComposition, is_valid_composition c → c.boys + c.girls = 35 := by
  sorry

end class_composition_theorem_l2739_273973


namespace jacket_cost_l2739_273915

theorem jacket_cost (total_spent : ℚ) (shorts_cost : ℚ) (shirt_cost : ℚ) 
  (h1 : total_spent = 33.56)
  (h2 : shorts_cost = 13.99)
  (h3 : shirt_cost = 12.14) :
  total_spent - (shorts_cost + shirt_cost) = 7.43 := by
  sorry

end jacket_cost_l2739_273915


namespace tony_fever_threshold_l2739_273946

/-- Calculates how many degrees above the fever threshold a person's temperature is -/
def degrees_above_fever_threshold (normal_temp fever_threshold temp_increase : ℝ) : ℝ :=
  (normal_temp + temp_increase) - fever_threshold

/-- Proves that Tony's temperature is 5 degrees above the fever threshold -/
theorem tony_fever_threshold :
  degrees_above_fever_threshold 95 100 10 = 5 := by
  sorry

end tony_fever_threshold_l2739_273946


namespace parallel_vectors_y_value_l2739_273961

/-- Given two parallel vectors a and b in R², where a = (1, 2) and b = (-2, y),
    prove that y must equal -4. -/
theorem parallel_vectors_y_value (y : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  y = -4 := by
sorry

end parallel_vectors_y_value_l2739_273961


namespace collinear_points_k_value_l2739_273955

/-- 
Given three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) in ℝ², 
this function returns true if they are collinear (lie on the same line).
-/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- 
Theorem: If the points (7, 10), (1, k), and (-8, 5) are collinear, then k = 40.
-/
theorem collinear_points_k_value : 
  collinear 7 10 1 k (-8) 5 → k = 40 := by
  sorry


end collinear_points_k_value_l2739_273955


namespace base_10_to_base_3_l2739_273926

theorem base_10_to_base_3 : 
  (2 * 3^6 + 0 * 3^5 + 0 * 3^4 + 1 * 3^3 + 1 * 3^2 + 2 * 3^1 + 2 * 3^0) = 1589 := by
  sorry

end base_10_to_base_3_l2739_273926


namespace sock_cost_calculation_l2739_273989

/-- The cost of each pair of socks that Niko bought --/
def sock_cost : ℝ := 2

/-- The number of pairs of socks Niko bought --/
def total_pairs : ℕ := 9

/-- The number of pairs Niko wants to sell with 25% profit --/
def pairs_with_percent_profit : ℕ := 4

/-- The number of pairs Niko wants to sell with $0.2 profit each --/
def pairs_with_fixed_profit : ℕ := 5

/-- The total profit Niko wants to make --/
def total_profit : ℝ := 3

/-- The profit percentage for the first group of socks --/
def profit_percentage : ℝ := 0.25

/-- The fixed profit amount for the second group of socks --/
def fixed_profit : ℝ := 0.2

theorem sock_cost_calculation :
  sock_cost * pairs_with_percent_profit * profit_percentage +
  pairs_with_fixed_profit * fixed_profit = total_profit ∧
  total_pairs = pairs_with_percent_profit + pairs_with_fixed_profit :=
by sorry

end sock_cost_calculation_l2739_273989


namespace kitten_growth_l2739_273907

/-- Given an initial length of a kitten and two doubling events, calculate the final length. -/
theorem kitten_growth (initial_length : ℝ) : 
  initial_length = 4 → (initial_length * 2 * 2 = 16) := by
  sorry

end kitten_growth_l2739_273907


namespace polar_to_rectangular_conversion_l2739_273970

/-- Conversion of polar coordinates to rectangular coordinates -/
theorem polar_to_rectangular_conversion
  (r : ℝ) (θ : ℝ) 
  (h : r = 10 ∧ θ = 3 * π / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2, 5 * Real.sqrt 2) := by
  sorry

#check polar_to_rectangular_conversion

end polar_to_rectangular_conversion_l2739_273970


namespace apple_box_problem_l2739_273912

theorem apple_box_problem (apples oranges : ℕ) : 
  oranges = 12 ∧ 
  (apples : ℝ) / (apples + (oranges - 6 : ℕ) : ℝ) = 0.7 → 
  apples = 14 := by
sorry

end apple_box_problem_l2739_273912


namespace no_valid_house_numbers_l2739_273900

def is_two_digit_prime (n : ℕ) : Prop :=
  10 < n ∧ n < 50 ∧ Nat.Prime n

def valid_house_number (w x y z : ℕ) : Prop :=
  w ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  is_two_digit_prime (w * 10 + x) ∧
  is_two_digit_prime (y * 10 + z) ∧
  (w * 10 + x) ≠ (y * 10 + z) ∧
  w + x + y + z = 19

theorem no_valid_house_numbers :
  ¬ ∃ w x y z : ℕ, valid_house_number w x y z :=
sorry

end no_valid_house_numbers_l2739_273900


namespace middle_person_height_l2739_273902

/-- Represents the heights of 5 people in a line -/
structure HeightLine where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  h₅ : ℝ
  height_order : h₁ ≤ h₂ ∧ h₂ ≤ h₃ ∧ h₃ ≤ h₄ ∧ h₄ ≤ h₅
  collinear_tops : ∃ (r : ℝ), r > 1 ∧ h₂ = h₁ * r ∧ h₃ = h₁ * r^2 ∧ h₄ = h₁ * r^3 ∧ h₅ = h₁ * r^4
  shortest_height : h₁ = 3
  tallest_height : h₅ = 7

/-- The height of the middle person in the line is √21 feet -/
theorem middle_person_height (line : HeightLine) : line.h₃ = Real.sqrt 21 := by
  sorry

end middle_person_height_l2739_273902


namespace f_inequality_l2739_273978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + a * x

theorem f_inequality (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) :
  (f a x₁ - f a x₂) / (x₂ - x₁) > 
  (Real.log ((x₁ + x₂) / 2) - a * ((x₁ + x₂) / 2) + a) :=
by sorry

end f_inequality_l2739_273978


namespace sequence_inequality_l2739_273994

theorem sequence_inequality (n : ℕ) (a : ℕ → ℚ) (h1 : a 0 = 1/2) 
  (h2 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
  sorry

end sequence_inequality_l2739_273994


namespace binomial_equation_unique_solution_l2739_273949

theorem binomial_equation_unique_solution :
  ∃! m : ℕ, (Nat.choose 23 m) + (Nat.choose 23 12) = (Nat.choose 24 13) ∧ m = 13 := by
  sorry

end binomial_equation_unique_solution_l2739_273949


namespace unique_triplet_solution_l2739_273996

theorem unique_triplet_solution :
  ∃! (m n p : ℕ+), 
    Nat.Prime p ∧ 
    (2 : ℕ)^(m : ℕ) * (p : ℕ)^2 + 1 = (n : ℕ)^5 ∧
    m = 1 ∧ n = 3 ∧ p = 11 := by
  sorry

end unique_triplet_solution_l2739_273996


namespace group_size_l2739_273941

theorem group_size (average_increase : ℝ) (weight_difference : ℝ) :
  average_increase = 3.5 →
  weight_difference = 28 →
  weight_difference = average_increase * 8 :=
by sorry

end group_size_l2739_273941


namespace unique_prime_triple_l2739_273908

theorem unique_prime_triple : 
  ∀ p q r : ℕ,
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r) →
  (Nat.Prime (4 * q - 1)) →
  ((p + q : ℚ) / (p + r) = r - p) →
  (p = 2 ∧ q = 3 ∧ r = 3) :=
by sorry

end unique_prime_triple_l2739_273908


namespace rectangle_area_change_l2739_273986

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 540) : 
  1.2 * L * (0.8 * W) = 518.4 := by
  sorry

end rectangle_area_change_l2739_273986


namespace decimal_difference_l2739_273962

/-- The value of the repeating decimal 0.717171... -/
def repeating_decimal : ℚ := 71 / 99

/-- The value of the terminating decimal 0.71 -/
def terminating_decimal : ℚ := 71 / 100

/-- Theorem stating that the difference between 0.717171... and 0.71 is 71/9900 -/
theorem decimal_difference : repeating_decimal - terminating_decimal = 71 / 9900 := by
  sorry

end decimal_difference_l2739_273962


namespace election_fourth_place_votes_l2739_273972

theorem election_fourth_place_votes :
  ∀ (total_votes : ℕ) (winner_votes : ℕ) (second_place_diff : ℕ) (third_place_diff : ℕ) (fourth_place_diff : ℕ),
    total_votes = 979 →
    winner_votes = second_place_diff + (winner_votes - second_place_diff) →
    winner_votes = third_place_diff + (winner_votes - third_place_diff) →
    winner_votes = fourth_place_diff + (winner_votes - fourth_place_diff) →
    second_place_diff = 53 →
    third_place_diff = 79 →
    fourth_place_diff = 105 →
    total_votes = winner_votes + (winner_votes - second_place_diff) + (winner_votes - third_place_diff) + (winner_votes - fourth_place_diff) →
    winner_votes - fourth_place_diff = 199 :=
by sorry

end election_fourth_place_votes_l2739_273972


namespace max_value_x_plus_y_l2739_273964

theorem max_value_x_plus_y (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 6 * y) :
  ∃ (max : ℝ), ∀ (x' y' : ℝ), 2 * x'^2 + 3 * y'^2 = 6 * y' → x' + y' ≤ max ∧ max = 1 + Real.sqrt 10 / 2 := by
  sorry

end max_value_x_plus_y_l2739_273964


namespace geese_percentage_among_non_swans_l2739_273901

theorem geese_percentage_among_non_swans 
  (geese_percent : ℝ) 
  (swans_percent : ℝ) 
  (herons_percent : ℝ) 
  (ducks_percent : ℝ) 
  (h1 : geese_percent = 30)
  (h2 : swans_percent = 25)
  (h3 : herons_percent = 10)
  (h4 : ducks_percent = 35)
  (h5 : geese_percent + swans_percent + herons_percent + ducks_percent = 100) :
  (geese_percent / (100 - swans_percent)) * 100 = 40 := by
  sorry

end geese_percentage_among_non_swans_l2739_273901


namespace not_in_first_quadrant_l2739_273984

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def FirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: If mn ≤ 0, then point A(m,n) cannot be in the first quadrant -/
theorem not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) :
  ¬FirstQuadrant ⟨m, n⟩ := by
  sorry


end not_in_first_quadrant_l2739_273984


namespace bottle_production_theorem_l2739_273991

/-- Given a number of machines and their production rate, calculate the total bottles produced in a given time -/
def bottlesProduced (numMachines : ℕ) (ratePerMinute : ℕ) (minutes : ℕ) : ℕ :=
  numMachines * ratePerMinute * minutes

/-- The production rate of a single machine -/
def singleMachineRate (totalMachines : ℕ) (totalRate : ℕ) : ℕ :=
  totalRate / totalMachines

theorem bottle_production_theorem (initialMachines : ℕ) (initialRate : ℕ) (newMachines : ℕ) (time : ℕ) :
  initialMachines = 6 →
  initialRate = 270 →
  newMachines = 14 →
  time = 4 →
  bottlesProduced newMachines (singleMachineRate initialMachines initialRate) time = 2520 :=
by
  sorry

end bottle_production_theorem_l2739_273991


namespace pet_shelter_adoption_time_l2739_273980

/-- Given an initial number of puppies, additional puppies, and a daily adoption rate,
    calculate the number of days required to adopt all puppies. -/
def days_to_adopt (initial : ℕ) (additional : ℕ) (adoption_rate : ℕ) : ℕ :=
  (initial + additional) / adoption_rate

/-- Theorem: For the given problem, it takes 2 days to adopt all puppies. -/
theorem pet_shelter_adoption_time : days_to_adopt 3 3 3 = 2 := by
  sorry

end pet_shelter_adoption_time_l2739_273980


namespace jill_watch_time_l2739_273920

/-- The total time Jill spent watching shows, given the length of the first show and a multiplier for the second show. -/
def total_watch_time (first_show_length : ℕ) (second_show_multiplier : ℕ) : ℕ :=
  first_show_length + first_show_length * second_show_multiplier

/-- Theorem stating that Jill spent 150 minutes watching shows. -/
theorem jill_watch_time : total_watch_time 30 4 = 150 := by
  sorry

end jill_watch_time_l2739_273920


namespace angle_sum_in_circle_l2739_273987

theorem angle_sum_in_circle (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + x = 360) → x = 24 := by
  sorry

end angle_sum_in_circle_l2739_273987


namespace triangle_existence_l2739_273979

/-- Represents a triangle with side lengths and angles -/
structure Triangle where
  a : ℝ  -- base length
  b : ℝ  -- one side length
  c : ℝ  -- other side length
  α : ℝ  -- angle opposite to side a
  β : ℝ  -- angle opposite to side b
  γ : ℝ  -- angle opposite to side c

/-- The existence of a triangle with given properties -/
theorem triangle_existence 
  (a : ℝ) 
  (bc_sum : ℝ) 
  (Δθ : ℝ) 
  (h_a_pos : a > 0) 
  (h_bc_sum_pos : bc_sum > 0) 
  (h_Δθ_range : 0 < Δθ ∧ Δθ < π) :
  ∃ (t : Triangle), 
    t.a = a ∧ 
    t.b + t.c = bc_sum ∧ 
    |t.β - t.γ| = Δθ ∧
    t.α + t.β + t.γ = π :=
  sorry

end triangle_existence_l2739_273979


namespace job_completion_time_l2739_273947

/-- Given that person A can complete a job in 18 days and both A and B together can complete it in 10 days, 
    this theorem proves that person B can complete the job alone in 22.5 days. -/
theorem job_completion_time (a_time b_time combined_time : ℝ) 
    (ha : a_time = 18)
    (hc : combined_time = 10)
    (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) :
    b_time = 22.5 := by
  sorry

end job_completion_time_l2739_273947


namespace cos_alpha_value_l2739_273918

theorem cos_alpha_value (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (α + Real.pi / 3) = -2/3) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 := by sorry

end cos_alpha_value_l2739_273918


namespace min_value_a5_plus_a6_l2739_273910

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

-- Define the theorem
theorem min_value_a5_plus_a6 (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
  ∃ (min : ℝ), min = 48 ∧ ∀ (a : ℕ → ℝ),
    is_positive_geometric_sequence a →
    a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
    a 5 + a 6 ≥ min :=
sorry

end min_value_a5_plus_a6_l2739_273910


namespace snowball_distance_l2739_273944

/-- The sum of an arithmetic sequence with first term 6, common difference 5, and 25 terms -/
def arithmetic_sum (first_term : ℕ) (common_diff : ℕ) (num_terms : ℕ) : ℕ :=
  (num_terms * (2 * first_term + (num_terms - 1) * common_diff)) / 2

/-- Theorem stating that the sum of the specific arithmetic sequence is 1650 -/
theorem snowball_distance : arithmetic_sum 6 5 25 = 1650 := by
  sorry

end snowball_distance_l2739_273944


namespace f_increasing_on_interval_l2739_273935

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end f_increasing_on_interval_l2739_273935


namespace fifth_selected_number_is_12_l2739_273952

-- Define the type for student numbers
def StudentNumber := Fin 50

-- Define the random number table as a list of natural numbers
def randomNumberTable : List ℕ :=
  [0627, 4313, 2432, 5327, 0941, 2512, 6317, 6323, 2616, 8045, 6011,
   1410, 9577, 7424, 6762, 4281, 1457, 2042, 5332, 3732, 2707, 3607,
   5124, 5179, 3014, 2310, 2118, 2191, 3726, 3890, 0140, 0523, 2617]

-- Define a function to check if a number is valid (between 01 and 50)
def isValidNumber (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 50

-- Define a function to select valid numbers from the table
def selectValidNumbers (table : List ℕ) : List StudentNumber :=
  sorry

-- State the theorem
theorem fifth_selected_number_is_12 :
  (selectValidNumbers randomNumberTable).nthLe 4 sorry = ⟨12, sorry⟩ :=
sorry

end fifth_selected_number_is_12_l2739_273952


namespace leonardo_chocolate_purchase_l2739_273928

theorem leonardo_chocolate_purchase (chocolate_cost : ℕ) (leonardo_money : ℕ) (borrowed_money : ℕ) : 
  chocolate_cost = 500 ∧ leonardo_money = 400 ∧ borrowed_money = 59 →
  chocolate_cost - (leonardo_money + borrowed_money) = 41 :=
by sorry

end leonardo_chocolate_purchase_l2739_273928


namespace roots_of_equation_l2739_273919

def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 2*x)*(x - 5)

theorem roots_of_equation : 
  {x : ℝ | f x = 0} = {0, 1, 2, 5} := by sorry

end roots_of_equation_l2739_273919


namespace cos_sq_plus_two_sin_double_l2739_273992

theorem cos_sq_plus_two_sin_double (α : Real) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := by
  sorry

end cos_sq_plus_two_sin_double_l2739_273992


namespace not_ripe_apples_l2739_273951

theorem not_ripe_apples (total : ℕ) (good : ℕ) (h1 : total = 14) (h2 : good = 8) :
  total - good = 6 := by
  sorry

end not_ripe_apples_l2739_273951


namespace f_increasing_on_negative_reals_l2739_273942

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 1

-- State the theorem
theorem f_increasing_on_negative_reals (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is an even function
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f m x < f m y) :=  -- f is increasing on (-∞, 0]
by
  sorry

end f_increasing_on_negative_reals_l2739_273942


namespace mango_selling_price_l2739_273909

theorem mango_selling_price 
  (loss_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (profit_price : ℝ) :
  loss_percentage = 20 →
  profit_percentage = 5 →
  profit_price = 6.5625 →
  ∃ (actual_price : ℝ), 
    actual_price = (1 - loss_percentage / 100) * (profit_price / (1 + profit_percentage / 100)) ∧
    actual_price = 5 := by
  sorry

end mango_selling_price_l2739_273909


namespace students_with_cat_and_dog_l2739_273983

theorem students_with_cat_and_dog (total : ℕ) (cat : ℕ) (dog : ℕ) (neither : ℕ) 
  (h1 : total = 28)
  (h2 : cat = 17)
  (h3 : dog = 10)
  (h4 : neither = 5)
  : ∃ both : ℕ, both = cat + dog - (total - neither) :=
by
  sorry

end students_with_cat_and_dog_l2739_273983


namespace compute_expression_l2739_273934

theorem compute_expression : 3 * 3^4 - 27^65 / 27^63 = -486 := by
  sorry

end compute_expression_l2739_273934


namespace ratio_of_sum_and_difference_l2739_273945

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end ratio_of_sum_and_difference_l2739_273945


namespace matrix_product_equality_l2739_273925

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 3, 1; 7, -1, 0; 0, 4, -2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -5, 2; 0, 4, 3; 1, 0, -1]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![3, 2, 12; 7, -39, 11; -2, 16, 14]

theorem matrix_product_equality : A * B = C := by sorry

end matrix_product_equality_l2739_273925


namespace min_value_expression_l2739_273988

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_prod : x * y * z = 2/3) :
  x^2 + 6*x*y + 18*y^2 + 12*y*z + 4*z^2 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ * y₀ * z₀ = 2/3 ∧
    x₀^2 + 6*x₀*y₀ + 18*y₀^2 + 12*y₀*z₀ + 4*z₀^2 = 18 :=
by sorry

end min_value_expression_l2739_273988


namespace plane_relationship_l2739_273948

-- Define the plane and line types
variable (Point : Type) (Vector : Type)
variable (Plane : Type) (Line : Type)

-- Define the containment relation
variable (contains : Plane → Line → Prop)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes
variable (intersect_planes : Plane → Plane → Prop)

-- Given conditions
variable (α β : Plane) (a b : Line)
variable (h1 : contains α a)
variable (h2 : contains β b)
variable (h3 : ¬ parallel_lines a b)

-- Theorem statement
theorem plane_relationship :
  parallel_planes α β ∨ intersect_planes α β :=
sorry

end plane_relationship_l2739_273948


namespace fifth_item_equals_one_fifteenth_l2739_273927

-- Define the sequence a_n
def a (n : ℕ) : ℚ := 2 / (n^2 + n : ℚ)

-- Theorem statement
theorem fifth_item_equals_one_fifteenth : a 5 = 1/15 := by
  sorry

end fifth_item_equals_one_fifteenth_l2739_273927


namespace tangent_line_at_2_l2739_273923

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

-- Theorem statement
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), 
    (∀ x y, y = m*x + b ↔ x - y - 4 = 0) ∧ 
    (m = f' 2) ∧
    (f 2 = m*2 + b) :=
sorry

end tangent_line_at_2_l2739_273923


namespace partnership_profit_calculation_l2739_273930

/-- A partnership business where one partner's investment and time are multiples of the other's -/
structure Partnership where
  investment_ratio : ℕ  -- Ratio of A's investment to B's
  time_ratio : ℕ        -- Ratio of A's investment time to B's
  b_profit : ℕ          -- B's profit in Rs

/-- Calculate the total profit of a partnership given B's profit -/
def total_profit (p : Partnership) : ℕ :=
  p.b_profit * (p.investment_ratio * p.time_ratio + 1)

/-- Theorem stating the total profit for the given partnership conditions -/
theorem partnership_profit_calculation (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.b_profit = 3000) :
  total_profit p = 21000 := by
  sorry

end partnership_profit_calculation_l2739_273930


namespace inequality_solution_set_l2739_273976

theorem inequality_solution_set (x : ℝ) :
  (|2*x - 3| < 1) ↔ (1 < x ∧ x < 2) :=
sorry

end inequality_solution_set_l2739_273976


namespace paramEquations_represent_line_l2739_273965

/-- Parametric equations of a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line y = 2x + 1 -/
def linearEquation (x y : ℝ) : Prop := y = 2 * x + 1

/-- The parametric equations x = t - 1 and y = 2t - 1 -/
def paramEquations : ParametricLine :=
  { x := fun t => t - 1
    y := fun t => 2 * t - 1 }

/-- Theorem: The parametric equations represent the line y = 2x + 1 -/
theorem paramEquations_represent_line :
  ∀ t : ℝ, linearEquation (paramEquations.x t) (paramEquations.y t) := by
  sorry


end paramEquations_represent_line_l2739_273965


namespace gcd_lcm_sum_ge_sum_l2739_273968

theorem gcd_lcm_sum_ge_sum (a b : ℕ+) : Nat.gcd a b + Nat.lcm a b ≥ a + b := by sorry

end gcd_lcm_sum_ge_sum_l2739_273968
