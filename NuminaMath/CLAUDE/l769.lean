import Mathlib

namespace average_weight_l769_76931

/-- Given three weights a, b, and c, prove that their average is 42 kg
    under the specified conditions. -/
theorem average_weight (a b c : ℝ) : 
  (a + b) / 2 = 40 →   -- The average weight of a and b is 40 kg
  (b + c) / 2 = 43 →   -- The average weight of b and c is 43 kg
  b = 40 →             -- The weight of b is 40 kg
  (a + b + c) / 3 = 42 -- The average weight of a, b, and c is 42 kg
  := by sorry

end average_weight_l769_76931


namespace three_digit_integer_with_specific_remainders_l769_76903

theorem three_digit_integer_with_specific_remainders :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
    n % 7 = 3 ∧ n % 8 = 6 ∧ n % 5 = 2 := by
  sorry

end three_digit_integer_with_specific_remainders_l769_76903


namespace percentage_male_worker_ants_l769_76927

theorem percentage_male_worker_ants (total_ants : ℕ) (female_worker_ants : ℕ) 
  (h1 : total_ants = 110)
  (h2 : female_worker_ants = 44) : 
  (((total_ants / 2 - female_worker_ants : ℚ) / (total_ants / 2)) * 100 = 20) := by
  sorry

end percentage_male_worker_ants_l769_76927


namespace starters_with_twin_l769_76936

def total_players : Nat := 12
def num_starters : Nat := 5
def num_twins : Nat := 2

theorem starters_with_twin (total_players num_starters num_twins : Nat) :
  total_players = 12 →
  num_starters = 5 →
  num_twins = 2 →
  (Nat.choose total_players num_starters) - (Nat.choose (total_players - num_twins) num_starters) = 540 := by
  sorry

end starters_with_twin_l769_76936


namespace fixed_order_queue_arrangement_l769_76955

def queue_arrangements (n : ℕ) (k : ℕ) : Prop :=
  n ≥ k ∧ (n - k).factorial * k.factorial * (n.choose k) = 20

theorem fixed_order_queue_arrangement :
  queue_arrangements 5 3 :=
sorry

end fixed_order_queue_arrangement_l769_76955


namespace parallel_to_same_line_implies_parallel_l769_76984

-- Define a type for lines in a plane
variable {Line : Type}

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Axiom: Parallel relation is symmetric
axiom parallel_symmetric {l1 l2 : Line} : parallel l1 l2 → parallel l2 l1

-- Axiom: Parallel relation is transitive
axiom parallel_transitive {l1 l2 l3 : Line} : parallel l1 l2 → parallel l2 l3 → parallel l1 l3

-- Theorem: If two lines are parallel to a third line, they are parallel to each other
theorem parallel_to_same_line_implies_parallel (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 :=
by sorry

end parallel_to_same_line_implies_parallel_l769_76984


namespace unique_square_divisible_by_five_l769_76917

theorem unique_square_divisible_by_five (y : ℕ) : 
  (∃ n : ℕ, y = n^2) ∧ 
  y % 5 = 0 ∧ 
  50 < y ∧ 
  y < 120 → 
  y = 100 := by
sorry

end unique_square_divisible_by_five_l769_76917


namespace quartic_roots_equivalence_l769_76916

theorem quartic_roots_equivalence (x : ℂ) : 
  (3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0) ↔ 
  (x + 1/x = (-1 + Real.sqrt 43)/3 ∨ x + 1/x = (-1 - Real.sqrt 43)/3) :=
by sorry

end quartic_roots_equivalence_l769_76916


namespace parallelogram_with_right_angle_is_rectangle_l769_76922

-- Define a parallelogram
structure Parallelogram :=
  (has_parallel_sides : Bool)

-- Define a rectangle
structure Rectangle extends Parallelogram :=
  (has_right_angle : Bool)

-- Theorem statement
theorem parallelogram_with_right_angle_is_rectangle 
  (p : Parallelogram) (h : Bool) : 
  (p.has_parallel_sides ∧ h) ↔ ∃ (r : Rectangle), r.has_right_angle ∧ r.has_parallel_sides = p.has_parallel_sides :=
sorry

end parallelogram_with_right_angle_is_rectangle_l769_76922


namespace travelers_checks_average_l769_76985

theorem travelers_checks_average (x y : ℕ) : 
  x + y = 30 →
  50 * x + 100 * y = 1800 →
  let remaining_50 := x - 6
  let remaining_100 := y
  let total_remaining := remaining_50 + remaining_100
  let total_value := 50 * remaining_50 + 100 * remaining_100
  (total_value : ℚ) / total_remaining = 125/2 := by sorry

end travelers_checks_average_l769_76985


namespace polynomial_from_root_relations_l769_76964

theorem polynomial_from_root_relations (α β γ : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 44*x - 46 = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∃ x₁ x₂ x₃ : ℝ, 
    α = x₁ + x₂ ∧ 
    β = x₁ + x₃ ∧ 
    γ = x₂ + x₃ ∧
    (∀ x, x^3 - 6*x^2 + 8*x - 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end polynomial_from_root_relations_l769_76964


namespace nested_subtraction_simplification_l769_76933

theorem nested_subtraction_simplification (x : ℝ) :
  1 - (2 - (3 - (4 - (5 - x)))) = 3 - x := by
  sorry

end nested_subtraction_simplification_l769_76933


namespace geometric_sequence_ratio_l769_76907

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 := by
  sorry

end geometric_sequence_ratio_l769_76907


namespace solution_set_when_a_is_neg_three_range_of_a_given_condition_l769_76909

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem for part 1
theorem solution_set_when_a_is_neg_three :
  {x : ℝ | f (-3) x ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem for part 2
theorem range_of_a_given_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end solution_set_when_a_is_neg_three_range_of_a_given_condition_l769_76909


namespace twelfth_term_of_specific_sequence_l769_76987

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

/-- The 12th term of the specific geometric sequence is 1/6561 -/
theorem twelfth_term_of_specific_sequence :
  geometric_sequence 27 (1/3) 12 = 1/6561 := by sorry

end twelfth_term_of_specific_sequence_l769_76987


namespace prime_pair_sum_cube_difference_l769_76947

theorem prime_pair_sum_cube_difference (p q : ℕ) : 
  Prime p ∧ Prime q ∧ p + q = (p - q)^3 → (p = 5 ∧ q = 3) := by
  sorry

end prime_pair_sum_cube_difference_l769_76947


namespace puppies_sold_l769_76960

/-- Given a pet store scenario, prove the number of puppies sold. -/
theorem puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) :
  initial_puppies ≥ puppies_per_cage * cages_used →
  initial_puppies - (puppies_per_cage * cages_used) =
    initial_puppies - puppies_per_cage * cages_used :=
by
  sorry

#check puppies_sold 102 9 9

end puppies_sold_l769_76960


namespace banana_price_is_60_cents_l769_76982

def apple_price : ℚ := 0.70
def total_cost : ℚ := 5.60
def total_fruits : ℕ := 9

theorem banana_price_is_60_cents :
  ∃ (num_apples num_bananas : ℕ) (banana_price : ℚ),
    num_apples + num_bananas = total_fruits ∧
    num_apples * apple_price + num_bananas * banana_price = total_cost ∧
    banana_price = 0.60 := by
  sorry

end banana_price_is_60_cents_l769_76982


namespace fraction_inequality_l769_76975

theorem fraction_inequality (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) :
  (b + m) / (a + m) < b / a := by
  sorry

end fraction_inequality_l769_76975


namespace tom_bought_three_decks_l769_76968

/-- The number of decks Tom bought -/
def tom_decks : ℕ := 3

/-- The cost of each deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Tom's friend bought -/
def friend_decks : ℕ := 5

/-- The total amount spent in dollars -/
def total_spent : ℕ := 64

/-- Theorem stating that Tom bought 3 decks given the conditions -/
theorem tom_bought_three_decks : 
  deck_cost * (tom_decks + friend_decks) = total_spent := by
  sorry

end tom_bought_three_decks_l769_76968


namespace hexagonal_glass_side_length_l769_76906

/-- A glass with regular hexagonal top and bottom, containing three identical spheres -/
structure HexagonalGlass where
  /-- Side length of the hexagonal bottom -/
  sideLength : ℝ
  /-- Volume of the glass -/
  volume : ℝ
  /-- The glass contains three identical spheres each touching every side -/
  spheresFit : True
  /-- The volume of the glass is 108 cm³ -/
  volumeIs108 : volume = 108

/-- The theorem stating the relationship between the glass volume and side length -/
theorem hexagonal_glass_side_length (g : HexagonalGlass) : 
  g.sideLength = 2 / Real.rpow 3 (1/3) :=
sorry

end hexagonal_glass_side_length_l769_76906


namespace sum_of_odd_coefficients_l769_76929

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀*x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆) →
  a₁ + a₃ + a₅ = -364 := by
sorry

end sum_of_odd_coefficients_l769_76929


namespace pebble_collection_sum_l769_76974

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem pebble_collection_sum : geometric_sum 2 2 10 = 2046 := by
  sorry

end pebble_collection_sum_l769_76974


namespace ink_cost_per_ml_l769_76998

/-- Proves that the cost of ink per milliliter is 50 cents given the specified conditions -/
theorem ink_cost_per_ml (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℕ) (total_cost : ℕ) : 
  num_classes = 5 → 
  boards_per_class = 2 → 
  ink_per_board = 20 → 
  total_cost = 100 → 
  (total_cost * 100) / (num_classes * boards_per_class * ink_per_board) = 50 := by
  sorry

#check ink_cost_per_ml

end ink_cost_per_ml_l769_76998


namespace circle_equation_l769_76912

/-- Given points A and B, and a circle whose center lies on a line, prove the equation of the circle. -/
theorem circle_equation (A B C : ℝ × ℝ) (r : ℝ) : 
  A = (1, -1) →
  B = (-1, 1) →
  C.1 + C.2 = 2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
  ∀ x y : ℝ, (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 1)^2 + (y - 1)^2 = 4 := by
  sorry

end circle_equation_l769_76912


namespace function_inequality_l769_76923

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f''(x) > f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv (deriv f)) x > f x)

-- State the theorem to be proved
theorem function_inequality : f (Real.log 2015) > 2015 * f 0 := by
  sorry

end function_inequality_l769_76923


namespace min_sum_bound_min_sum_achievable_l769_76994

theorem min_sum_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (6 * a) ≥ 3 / Real.rpow 90 (1/3) :=
sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (3 * b) + b / (5 * c) + c / (6 * a) = 3 / Real.rpow 90 (1/3) :=
sorry

end min_sum_bound_min_sum_achievable_l769_76994


namespace pet_store_combinations_l769_76946

def num_puppies : ℕ := 10
def num_kittens : ℕ := 8
def num_hamsters : ℕ := 12
def num_rabbits : ℕ := 4

def alice_choice : ℕ := num_puppies + num_rabbits

/-- The number of ways Alice, Bob, Charlie, and Dana can buy pets and leave the store satisfied. -/
theorem pet_store_combinations : ℕ := by
  sorry

end pet_store_combinations_l769_76946


namespace quadrilateral_area_is_0125_l769_76995

/-- The area of the quadrilateral formed by the intersection of four lines -/
def quadrilateral_area (line1 line2 : ℝ → ℝ → Prop) (x_line y_line : ℝ → Prop) : ℝ := sorry

/-- The first line: 3x + 4y - 12 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

/-- The second line: 5x - 4y - 10 = 0 -/
def line2 (x y : ℝ) : Prop := 5 * x - 4 * y - 10 = 0

/-- The vertical line: x = 3 -/
def x_line (x : ℝ) : Prop := x = 3

/-- The horizontal line: y = 1 -/
def y_line (y : ℝ) : Prop := y = 1

theorem quadrilateral_area_is_0125 : 
  quadrilateral_area line1 line2 x_line y_line = 0.125 := by sorry

end quadrilateral_area_is_0125_l769_76995


namespace trigonometric_system_solution_l769_76992

theorem trigonometric_system_solution (x y : ℝ) (k : ℤ) :
  (Real.cos x)^2 + (Real.cos y)^2 = 0.25 →
  x + y = 5 * Real.pi / 6 →
  ((x = Real.pi / 2 * (2 * ↑k + 1) ∧ y = Real.pi / 3 * (1 - 3 * ↑k)) ∨
   (x = Real.pi / 3 * (3 * ↑k + 1) ∧ y = Real.pi / 2 * (1 - 2 * ↑k))) :=
by sorry

end trigonometric_system_solution_l769_76992


namespace marbles_selection_with_red_l769_76971

def total_marbles : ℕ := 10
def marbles_to_choose : ℕ := 5

theorem marbles_selection_with_red (total : ℕ) (choose : ℕ) 
  (h1 : total = total_marbles) 
  (h2 : choose = marbles_to_choose) 
  (h3 : total > 0) 
  (h4 : choose > 0) 
  (h5 : total ≥ choose) :
  Nat.choose total choose - Nat.choose (total - 1) choose = 126 := by
  sorry

end marbles_selection_with_red_l769_76971


namespace infinite_solution_equation_non_solutions_l769_76930

/-- Given an equation with infinitely many solutions, prove the number and sum of non-solutions -/
theorem infinite_solution_equation_non_solutions (A B C : ℚ) : 
  (∀ x, (x + B) * (A * x + 42) = 3 * (x + C) * (x + 9)) →
  (∃! s : Finset ℚ, s.card = 2 ∧ 
    (∀ x ∈ s, (x + B) * (A * x + 42) ≠ 3 * (x + C) * (x + 9)) ∧
    (∀ x ∉ s, (x + B) * (A * x + 42) = 3 * (x + C) * (x + 9)) ∧
    s.sum id = -187/13) :=
by sorry

end infinite_solution_equation_non_solutions_l769_76930


namespace sum_of_possible_x_minus_y_values_l769_76951

theorem sum_of_possible_x_minus_y_values (x y : ℝ) 
  (eq1 : x^2 - x*y + x = 2018)
  (eq2 : y^2 - x*y - y = 52) : 
  ∃ (z₁ z₂ : ℝ), (z₁ = x - y ∨ z₂ = x - y) ∧ z₁ + z₂ = -1 := by
sorry

end sum_of_possible_x_minus_y_values_l769_76951


namespace area_BQW_is_48_l769_76973

/-- Rectangle ABCD with specific measurements and areas -/
structure Rectangle where
  AB : ℝ
  AZ : ℝ
  WC : ℝ
  area_ZWCD : ℝ
  h : AB = 16
  h' : AZ = 8
  h'' : WC = 8
  h''' : area_ZWCD = 160

/-- The area of triangle BQW in the given rectangle -/
def area_BQW (r : Rectangle) : ℝ := 48

/-- Theorem stating that the area of triangle BQW is 48 square units -/
theorem area_BQW_is_48 (r : Rectangle) : area_BQW r = 48 := by
  sorry

end area_BQW_is_48_l769_76973


namespace compound_proposition_falsehood_l769_76932

theorem compound_proposition_falsehood (p q : Prop) : 
  ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end compound_proposition_falsehood_l769_76932


namespace problem_solution_l769_76910

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem problem_solution (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (3*x + 10) = f (3*x + 1))
  (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 := by
  sorry


end problem_solution_l769_76910


namespace max_area_folded_rectangle_l769_76944

/-- Given a rectangle ABCD with perimeter 24 and AB > AD, when folded along its diagonal AC
    such that AB meets DC at point P, the maximum area of triangle ADP is 72√2. -/
theorem max_area_folded_rectangle (AB AD : ℝ) (h1 : AB > AD) (h2 : AB + AD = 12) :
  let x := AB
  let a := (x^2 - 12*x + 72) / x
  let DP := (12*x - 72) / x
  let area := 3 * (12 - x) * ((12*x - 72) / x)
  ∃ (max_area : ℝ), (∀ x, 0 < x → x < 12 → area ≤ max_area) ∧ max_area = 72 * Real.sqrt 2 :=
sorry

end max_area_folded_rectangle_l769_76944


namespace joyce_apples_l769_76963

/-- The number of apples Joyce gave to Larry -/
def apples_given : ℕ := 52

/-- The number of apples Joyce had left -/
def apples_left : ℕ := 23

/-- The total number of apples Joyce started with -/
def initial_apples : ℕ := apples_given + apples_left

theorem joyce_apples : initial_apples = 75 := by
  sorry

end joyce_apples_l769_76963


namespace h_has_one_zero_and_inequality_l769_76915

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := Real.exp x - 1
noncomputable def h (x : ℝ) := f x - g x

theorem h_has_one_zero_and_inequality :
  (∃! x, h x = 0) ∧
  (g (Real.exp 2 - Real.log 2 - 1) > Real.log (Real.exp 2 - Real.log 2)) ∧
  (Real.log (Real.exp 2 - Real.log 2) > 2 - f (Real.log 2)) := by
  sorry

end h_has_one_zero_and_inequality_l769_76915


namespace perpendicular_transitivity_l769_76900

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the statement to be proved
theorem perpendicular_transitivity 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) (h2 : α ≠ β)
  (h3 : perp a α) (h4 : perp a β) (h5 : perp b β) :
  perp b α :=
sorry

end perpendicular_transitivity_l769_76900


namespace simplify_sqrt_m_squared_n_l769_76940

theorem simplify_sqrt_m_squared_n
  (m n : ℝ)
  (h1 : m < 0)
  (h2 : m^2 * n ≥ 0) :
  Real.sqrt (m^2 * n) = -m * Real.sqrt n :=
by sorry

end simplify_sqrt_m_squared_n_l769_76940


namespace intersection_range_l769_76948

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = 4*x + m

-- Define symmetry with respect to the line
def symmetric_points (x1 y1 x2 y2 m : ℝ) : Prop :=
  line ((x1 + x2)/2) ((y1 + y2)/2) m

-- Theorem statement
theorem intersection_range (m : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    ellipse x1 y1 ∧ 
    ellipse x2 y2 ∧ 
    line x1 y1 m ∧ 
    line x2 y2 m ∧ 
    symmetric_points x1 y1 x2 y2 m) ↔ 
  -2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13 :=
sorry

end intersection_range_l769_76948


namespace accident_rate_calculation_l769_76956

theorem accident_rate_calculation (total_vehicles : ℕ) (accident_vehicles : ℕ) 
  (rate_vehicles : ℕ) (rate_accidents : ℕ) :
  total_vehicles = 3000000000 →
  accident_vehicles = 2880 →
  rate_accidents = 96 →
  (rate_accidents : ℚ) / rate_vehicles = (accident_vehicles : ℚ) / total_vehicles →
  rate_vehicles = 100000000 := by
sorry

end accident_rate_calculation_l769_76956


namespace parallelogram_height_base_difference_l769_76970

theorem parallelogram_height_base_difference 
  (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 24) 
  (h_base : base = 4) 
  (h_parallelogram : area = base * height) : 
  height - base = 2 := by
sorry

end parallelogram_height_base_difference_l769_76970


namespace closest_ratio_to_one_l769_76962

/-- Represents the admission fee structure and total collection --/
structure AdmissionData where
  adult_fee : ℕ
  child_fee : ℕ
  total_collection : ℕ

/-- Represents a valid combination of adults and children --/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Checks if the given attendance satisfies the total collection --/
def is_valid_attendance (data : AdmissionData) (att : Attendance) : Prop :=
  data.adult_fee * att.adults + data.child_fee * att.children = data.total_collection

/-- Calculates the absolute difference between the ratio and 1 --/
def ratio_diff_from_one (att : Attendance) : ℚ :=
  |att.adults / att.children - 1|

theorem closest_ratio_to_one (data : AdmissionData) :
  data.adult_fee = 25 →
  data.child_fee = 12 →
  data.total_collection = 1950 →
  ∃ (best : Attendance),
    is_valid_attendance data best ∧
    best.adults > 0 ∧
    best.children > 0 ∧
    ∀ (att : Attendance),
      is_valid_attendance data att →
      att.adults > 0 →
      att.children > 0 →
      ratio_diff_from_one best ≤ ratio_diff_from_one att ∧
      (best.adults = 54 ∧ best.children = 50) :=
sorry

end closest_ratio_to_one_l769_76962


namespace train_crossing_time_l769_76941

/-- Represents the time it takes for a train to cross a tree given its length and the time it takes to pass a platform of known length. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 1200)
  (h2 : platform_length = 1000)
  (h3 : platform_crossing_time = 220) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 120 := by
  sorry

#check train_crossing_time

end train_crossing_time_l769_76941


namespace sphere_radius_from_surface_area_l769_76983

theorem sphere_radius_from_surface_area :
  ∀ (r : ℝ), (4 : ℝ) * Real.pi * r^2 = (4 : ℝ) * Real.pi → r = 1 := by
  sorry

end sphere_radius_from_surface_area_l769_76983


namespace sequence_is_constant_l769_76996

/-- A sequence of positive integers -/
def Sequence := ℕ → ℕ

/-- The divisibility condition for the sequence -/
def DivisibilityCondition (a : Sequence) : Prop :=
  ∀ i j : ℕ, i > j → (i - j)^(2*(i - j)) + 1 ∣ a i - a j

/-- The theorem stating that a sequence satisfying the divisibility condition is constant -/
theorem sequence_is_constant (a : Sequence) (h : DivisibilityCondition a) :
  ∀ n m : ℕ, a n = a m :=
sorry

end sequence_is_constant_l769_76996


namespace set_equals_interval_l769_76939

-- Define the set {x | x ≥ 2}
def S : Set ℝ := {x : ℝ | x ≥ 2}

-- Define the interval [2, +∞)
def I : Set ℝ := Set.Ici 2

-- Theorem stating that S is equal to I
theorem set_equals_interval : S = I := by sorry

end set_equals_interval_l769_76939


namespace system_solution_l769_76959

/-- The system of differential equations -/
def system (t x y : ℝ) : Prop :=
  ∃ (dt dx dy : ℝ), dt / (4*y - 5*x) = dx / (5*t - 3*y) ∧ dx / (5*t - 3*y) = dy / (3*x - 4*t)

/-- The general solution of the system -/
def solution (t x y : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ), 3*t + 4*x + 5*y = C₁ ∧ t^2 + x^2 + y^2 = C₂

/-- Theorem stating that the solution satisfies the system -/
theorem system_solution :
  ∀ (t x y : ℝ), system t x y → solution t x y :=
sorry

end system_solution_l769_76959


namespace coordinate_points_count_l769_76991

theorem coordinate_points_count (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  Finset.card (Finset.product S S) = 25 := by
  sorry

end coordinate_points_count_l769_76991


namespace cos_120_degrees_l769_76969

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end cos_120_degrees_l769_76969


namespace abc_inequality_l769_76905

theorem abc_inequality (a b c : Real) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq_a : Real.exp a = 9 * a * Real.log 11)
  (eq_b : Real.exp b = 10 * b * Real.log 10)
  (eq_c : Real.exp c = 11 * c * Real.log 9) :
  c > b ∧ b > a := by sorry

end abc_inequality_l769_76905


namespace sqrt_two_subtraction_l769_76961

theorem sqrt_two_subtraction : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_two_subtraction_l769_76961


namespace exists_function_satisfying_properties_l769_76926

/-- A strictly increasing function from natural numbers to natural numbers -/
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → f m < f n

/-- The property that f(f(f(n))) = n + 2f(n) for all n -/
def TripleCompositionProperty (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f (f n)) = n + 2 * (f n)

/-- The main theorem stating the existence of a function satisfying both properties -/
theorem exists_function_satisfying_properties :
  ∃ f : ℕ → ℕ, StrictlyIncreasing f ∧ TripleCompositionProperty f :=
sorry

end exists_function_satisfying_properties_l769_76926


namespace cube_volume_surface_area_l769_76967

theorem cube_volume_surface_area (V : ℝ) : 
  (∃ (x : ℝ), V = x^3 ∧ 2*V = 6*x^2) → V = 27 := by
  sorry

end cube_volume_surface_area_l769_76967


namespace cycle_loss_percentage_l769_76993

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

theorem cycle_loss_percentage :
  let costPrice : ℚ := 1400
  let sellingPrice : ℚ := 1330
  percentageLoss costPrice sellingPrice = 5 := by
  sorry

end cycle_loss_percentage_l769_76993


namespace quadratic_increasing_implies_a_bound_l769_76943

/-- A quadratic function f(x) = x^2 + bx + c with b = 2a-1 -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := λ x => x^2 + (2*a - 1)*x + 3

/-- The function is increasing on the interval (1, +∞) -/
def IsIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x < f y

theorem quadratic_increasing_implies_a_bound (a : ℝ) :
  IsIncreasingOn (QuadraticFunction a) → a ≥ -1/2 := by
  sorry

end quadratic_increasing_implies_a_bound_l769_76943


namespace f_of_g_5_l769_76911

def g (x : ℝ) : ℝ := 3 * x - 4

def f (x : ℝ) : ℝ := 2 * x + 5

theorem f_of_g_5 : f (g 5) = 27 := by
  sorry

end f_of_g_5_l769_76911


namespace safari_count_l769_76913

theorem safari_count (antelopes : ℕ) (h1 : antelopes = 80) : ∃ (rabbits hyenas wild_dogs leopards giraffes lions elephants zebras hippos : ℕ),
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards * 2 = rabbits ∧
  giraffes = antelopes + 15 ∧
  lions = leopards + giraffes ∧
  elephants = 3 * lions ∧
  4 * zebras = 3 * antelopes ∧
  hippos = zebras + zebras / 10 ∧
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants + zebras + hippos = 1334 :=
by
  sorry


end safari_count_l769_76913


namespace sequence_sum_theorem_l769_76924

/-- Geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Arithmetic sequence with the given property -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Sum of first n terms of a sequence -/
def sum_of_terms (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

theorem sequence_sum_theorem (a b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  3 * a 5 - a 3 * a 7 = 0 →
  b 5 = a 5 →
  sum_of_terms b 9 = 27 := by
  sorry

end sequence_sum_theorem_l769_76924


namespace soft_drink_cost_l769_76901

/-- The cost of a 12-pack of soft drinks in dollars -/
def pack_cost : ℚ := 299 / 100

/-- The number of cans in a pack -/
def cans_per_pack : ℕ := 12

/-- The cost per can of soft drink -/
def cost_per_can : ℚ := pack_cost / cans_per_pack

/-- Rounding function to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ := round (100 * x) / 100

theorem soft_drink_cost :
  round_to_cent cost_per_can = 25 / 100 :=
sorry

end soft_drink_cost_l769_76901


namespace cos_five_pi_sixths_l769_76952

theorem cos_five_pi_sixths : Real.cos (5 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_five_pi_sixths_l769_76952


namespace zero_in_interval_l769_76914

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) ∧ f x = 0 :=
by
  have h1 : f (1/4) < 0 := by sorry
  have h2 : f (1/2) > 0 := by sorry
  sorry

end zero_in_interval_l769_76914


namespace unique_function_theorem_l769_76966

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ (m n : ℕ) (p : ℕ), Nat.Prime p →
    (p ∣ f (m + n) ↔ p ∣ (f m + f n))

theorem unique_function_theorem :
  ∃! f : ℕ → ℕ, is_surjective f ∧ satisfies_condition f :=
by
  sorry

end unique_function_theorem_l769_76966


namespace susannah_swims_24_times_l769_76999

/-- The number of times Camden went swimming in March -/
def camden_swims : ℕ := 16

/-- The number of weeks in March -/
def weeks_in_march : ℕ := 4

/-- The number of times Camden swam per week -/
def camden_swims_per_week : ℕ := camden_swims / weeks_in_march

/-- The number of additional times Susannah swam per week compared to Camden -/
def susannah_additional_swims : ℕ := 2

/-- The number of times Susannah swam per week -/
def susannah_swims_per_week : ℕ := camden_swims_per_week + susannah_additional_swims

/-- The total number of times Susannah went swimming in March -/
def susannah_total_swims : ℕ := susannah_swims_per_week * weeks_in_march

theorem susannah_swims_24_times : susannah_total_swims = 24 := by
  sorry

end susannah_swims_24_times_l769_76999


namespace f_properties_l769_76997

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 1) - a

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x = f a x → x ∈ {x : ℝ | x < -1 ∨ x > -1}) ∧
  (∀ x, f a x = -f a (-x - 2)) :=
by sorry

end f_properties_l769_76997


namespace intersecting_line_theorem_l769_76979

/-- A line passing through (a, 0) intersecting y^2 = 4x at P and Q -/
structure IntersectingLine (a : ℝ) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  line_through_a : P.2 / (P.1 - a) = Q.2 / (Q.1 - a)
  P_on_parabola : P.2^2 = 4 * P.1
  Q_on_parabola : Q.2^2 = 4 * Q.1

/-- The reciprocal sum of squared distances is constant -/
def constant_sum (a : ℝ) :=
  ∃ (k : ℝ), ∀ (l : IntersectingLine a),
    1 / ((l.P.1 - a)^2 + l.P.2^2) + 1 / ((l.Q.1 - a)^2 + l.Q.2^2) = k

/-- If the reciprocal sum of squared distances is constant, then a = 2 -/
theorem intersecting_line_theorem :
  ∀ a : ℝ, constant_sum a → a = 2 := by sorry

end intersecting_line_theorem_l769_76979


namespace problem_solution_l769_76920

theorem problem_solution (a b : ℕ+) 
  (sum_constraint : a + b = 30)
  (equation_constraint : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end problem_solution_l769_76920


namespace golden_retriever_adult_weight_l769_76935

/-- Represents the weight of a golden retriever at different stages of growth -/
structure DogWeight where
  initial : ℕ  -- Weight at 7 weeks
  week9 : ℕ    -- Weight at 9 weeks
  month3 : ℕ   -- Weight at 3 months
  month5 : ℕ   -- Weight at 5 months
  adult : ℕ    -- Adult weight at 1 year

/-- Calculates the adult weight of a golden retriever based on its growth pattern -/
def calculateAdultWeight (w : DogWeight) : ℕ :=
  w.initial * 2 * 2 * 2 + 30

/-- Theorem stating that the adult weight of the golden retriever is 78 pounds -/
theorem golden_retriever_adult_weight (w : DogWeight) 
  (h1 : w.initial = 6)
  (h2 : w.week9 = w.initial * 2)
  (h3 : w.month3 = w.week9 * 2)
  (h4 : w.month5 = w.month3 * 2)
  (h5 : w.adult = w.month5 + 30) :
  w.adult = 78 := by
  sorry


end golden_retriever_adult_weight_l769_76935


namespace two_roots_iff_a_eq_twenty_l769_76904

/-- The quadratic equation in x parametrized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 2) + a * (39 - 20*x) + 20

/-- The condition for at least two distinct roots -/
def has_at_least_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- The main theorem -/
theorem two_roots_iff_a_eq_twenty :
  ∀ a : ℝ, has_at_least_two_distinct_roots a ↔ a = 20 := by sorry

end two_roots_iff_a_eq_twenty_l769_76904


namespace ping_pong_games_l769_76950

theorem ping_pong_games (total_games : ℕ) (frankie_games carla_games : ℕ) : 
  total_games = 30 →
  frankie_games + carla_games = total_games →
  frankie_games = carla_games / 2 →
  carla_games = 20 := by
sorry

end ping_pong_games_l769_76950


namespace cameron_chase_speed_ratio_l769_76990

/-- Proves that the ratio of Cameron's speed to Chase's speed is 2:1 given the conditions -/
theorem cameron_chase_speed_ratio 
  (cameron_speed chase_speed danielle_speed : ℝ)
  (danielle_time chase_time : ℝ)
  (h1 : danielle_speed = 3 * cameron_speed)
  (h2 : danielle_time = 30)
  (h3 : chase_time = 180)
  (h4 : danielle_speed * danielle_time = chase_speed * chase_time) :
  cameron_speed / chase_speed = 2 := by
  sorry

end cameron_chase_speed_ratio_l769_76990


namespace circle_area_equilateral_triangle_l769_76919

theorem circle_area_equilateral_triangle (s : ℝ) (h : s = 12) :
  let R := s / Real.sqrt 3
  (π * R^2) = 48 * π := by
  sorry

end circle_area_equilateral_triangle_l769_76919


namespace second_sum_is_1720_l769_76972

/-- Given a total sum of 2795 rupees divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is equal to 1720 rupees. -/
theorem second_sum_is_1720 (total : ℝ) (first_part second_part : ℝ) : 
  total = 2795 →
  first_part + second_part = total →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1720 := by
  sorry

end second_sum_is_1720_l769_76972


namespace product_odd_even_is_odd_l769_76986

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem product_odd_even_is_odd (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsEven g) :
  IsOdd (fun x ↦ f x * g x) := by
  sorry


end product_odd_even_is_odd_l769_76986


namespace exactly_one_correct_l769_76934

/-- Represents a geometric statement --/
inductive GeometricStatement
  | complement_acute : GeometricStatement
  | equal_vertical : GeometricStatement
  | unique_parallel : GeometricStatement
  | perpendicular_distance : GeometricStatement
  | corresponding_angles : GeometricStatement

/-- Checks if a geometric statement is correct --/
def is_correct (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.complement_acute => True
  | _ => False

/-- The list of all geometric statements --/
def all_statements : List GeometricStatement :=
  [GeometricStatement.complement_acute,
   GeometricStatement.equal_vertical,
   GeometricStatement.unique_parallel,
   GeometricStatement.perpendicular_distance,
   GeometricStatement.corresponding_angles]

/-- Theorem stating that exactly one statement is correct --/
theorem exactly_one_correct :
  ∃! (s : GeometricStatement), s ∈ all_statements ∧ is_correct s :=
sorry

end exactly_one_correct_l769_76934


namespace sum_equidistant_terms_l769_76937

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_equidistant_terms 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a7 : a 7 = 12) : 
  a 2 + a 12 = 24 := by
sorry

end sum_equidistant_terms_l769_76937


namespace solve_system_l769_76921

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end solve_system_l769_76921


namespace opposite_sqrt_81_l769_76978

theorem opposite_sqrt_81 : -(Real.sqrt 81) = -9 := by
  sorry

end opposite_sqrt_81_l769_76978


namespace diamond_ratio_l769_76977

def diamond (n m : ℝ) : ℝ := n^4 * m^3

theorem diamond_ratio : (diamond 3 2) / (diamond 2 3) = 3/2 := by
  sorry

end diamond_ratio_l769_76977


namespace surface_area_specific_cube_l769_76989

/-- Calculates the surface area of a cube with holes -/
def surface_area_cube_with_holes (cube_edge_length : ℝ) (hole_side_length : ℝ) (num_holes_per_face : ℕ) : ℝ :=
  let original_surface_area := 6 * cube_edge_length^2
  let area_removed_by_holes := 6 * num_holes_per_face * hole_side_length^2
  let area_exposed_by_holes := 6 * num_holes_per_face * 4 * hole_side_length^2
  original_surface_area - area_removed_by_holes + area_exposed_by_holes

/-- Theorem stating the surface area of the specific cube with holes -/
theorem surface_area_specific_cube : surface_area_cube_with_holes 4 1 2 = 132 := by
  sorry

end surface_area_specific_cube_l769_76989


namespace inscribed_rectangle_area_l769_76980

/-- The area of a rectangle inscribed in a triangle -/
theorem inscribed_rectangle_area (b h x : ℝ) (hb : b > 0) (hh : h > 0) (hx : x > 0) (hxh : x < h) :
  let triangle_area := (1/2) * b * h
  let rectangle_base := b * (1 - x/h)
  let rectangle_area := x * rectangle_base
  rectangle_area = (b * x / h) * (h - x) :=
by sorry

end inscribed_rectangle_area_l769_76980


namespace triangle_angle_measure_l769_76945

theorem triangle_angle_measure (D E F : ℝ) : 
  -- DEF is a triangle
  D + E + F = 180 →
  -- Measure of angle E is three times the measure of angle F
  E = 3 * F →
  -- Angle F is 15°
  F = 15 →
  -- Then the measure of angle D is 120°
  D = 120 := by
sorry

end triangle_angle_measure_l769_76945


namespace seventh_term_is_eleven_l769_76954

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first five terms is 35 -/
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 35
  /-- The sixth term is 10 -/
  sixth_term : a + 5*d = 10

/-- The seventh term of the arithmetic sequence is 11 -/
theorem seventh_term_is_eleven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 11 := by
  sorry

end seventh_term_is_eleven_l769_76954


namespace circle_center_and_radius_circle_properties_l769_76918

theorem circle_center_and_radius 
  (x y : ℝ) : 
  x^2 + y^2 + 4*x - 6*y = 11 ↔ 
  (x + 2)^2 + (y - 3)^2 = 24 :=
by sorry

theorem circle_properties : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (-2, 3) ∧ 
  radius = 2 * Real.sqrt 6 ∧
  ∀ (x y : ℝ), x^2 + y^2 + 4*x - 6*y = 11 ↔ 
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_circle_properties_l769_76918


namespace right_triangle_perimeter_l769_76953

/-- Given a right triangle with area 180 square units and one leg of length 18 units,
    its perimeter is 38 + 2√181 units. -/
theorem right_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 18 →
  a^2 + b^2 = c^2 →
  a + b + c = 38 + 2 * Real.sqrt 181 := by
  sorry

end right_triangle_perimeter_l769_76953


namespace closer_to_one_than_four_closer_to_zero_than_ax_l769_76902

-- Part 1
theorem closer_to_one_than_four (x : ℝ) :
  |x^2 - 1| < |4 - 1| → x ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

-- Part 2
theorem closer_to_zero_than_ax (x a : ℝ) :
  a > 0 → |x^2 + a| < |(a + 1) * x| →
  (0 < a ∧ a < 1 → x ∈ Set.Ioo (-1 : ℝ) (-a) ∪ Set.Ioo a 1) ∧
  (a = 1 → False) ∧
  (a > 1 → x ∈ Set.Ioo (-a : ℝ) (-1) ∪ Set.Ioo 1 a) :=
sorry

end closer_to_one_than_four_closer_to_zero_than_ax_l769_76902


namespace function_equality_condition_l769_76949

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem function_equality_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a ∈ Set.Ioi (2/3) ∪ Set.Iic 0 :=
sorry

end function_equality_condition_l769_76949


namespace negation_of_existence_negation_of_ln_positive_l769_76928

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem negation_of_ln_positive :
  (¬ ∃ x > 0, Real.log x > 0) ↔ (∀ x > 0, Real.log x ≤ 0) := by sorry

end negation_of_existence_negation_of_ln_positive_l769_76928


namespace arcsin_of_neg_one_l769_76965

theorem arcsin_of_neg_one : Real.arcsin (-1) = -π / 2 := by sorry

end arcsin_of_neg_one_l769_76965


namespace horner_method_v3_l769_76908

def horner_polynomial (x : ℤ) : ℤ := 12 + 35*x - 8*x^2 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v0 : ℤ := 3
def horner_v1 (x : ℤ) : ℤ := horner_v0 * x + 5
def horner_v2 (x : ℤ) : ℤ := horner_v1 x * x + 6
def horner_v3 (x : ℤ) : ℤ := horner_v2 x * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
sorry

end horner_method_v3_l769_76908


namespace purely_imaginary_complex_number_l769_76976

/-- If the complex number lg(m^2-2m-2) + (m^2+3m+2)i is purely imaginary and m is real, then m = 3 -/
theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)).im ≠ 0 ∧ 
  (Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)).re = 0 → 
  m = 3 := by sorry

end purely_imaginary_complex_number_l769_76976


namespace rectangle_ratio_l769_76957

theorem rectangle_ratio (l w : ℝ) (hl : l = 10) (hp : 2 * l + 2 * w = 36) :
  w / l = 4 / 5 := by
  sorry

end rectangle_ratio_l769_76957


namespace possible_d_values_l769_76925

theorem possible_d_values : 
  ∀ d : ℤ, (∃ e f : ℤ, ∀ x : ℤ, (x - d) * (x - 12) + 1 = (x + e) * (x + f)) → (d = 22 ∨ d = 26) :=
by sorry

end possible_d_values_l769_76925


namespace two_layer_wallpaper_area_l769_76938

/-- Given the total wallpaper area, wall area, and area covered by three layers,
    calculate the area covered by exactly two layers of wallpaper. -/
theorem two_layer_wallpaper_area
  (total_area : ℝ)
  (wall_area : ℝ)
  (three_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : wall_area = 180)
  (h3 : three_layer_area = 40) :
  total_area - wall_area - three_layer_area = 80 := by
sorry

end two_layer_wallpaper_area_l769_76938


namespace lanas_tickets_l769_76988

/-- The number of tickets Lana bought for herself and friends -/
def tickets_for_friends : ℕ := sorry

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 6

/-- The number of extra tickets Lana bought -/
def extra_tickets : ℕ := 2

/-- The total amount Lana spent in dollars -/
def total_spent : ℕ := 60

theorem lanas_tickets : 
  (tickets_for_friends + extra_tickets) * ticket_cost = total_spent ∧ 
  tickets_for_friends = 8 := by
  sorry

end lanas_tickets_l769_76988


namespace field_area_theorem_l769_76942

/-- Represents a rectangular field with a given length and breadth. -/
structure RectangularField where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular field. -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.length + field.breadth)

/-- Calculates the area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.length * field.breadth

/-- Theorem: The area of a rectangular field with breadth 60% of its length
    and perimeter 800 m is 37500 square meters. -/
theorem field_area_theorem :
  ∃ (field : RectangularField),
    field.breadth = 0.6 * field.length ∧
    perimeter field = 800 ∧
    area field = 37500 := by
  sorry

end field_area_theorem_l769_76942


namespace season_games_count_l769_76958

/-- The number of teams in the sports conference -/
def total_teams : ℕ := 16

/-- The number of divisions in the sports conference -/
def num_divisions : ℕ := 2

/-- The number of teams in each division -/
def teams_per_division : ℕ := 8

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- The total number of games in a complete season -/
def total_games : ℕ := 296

theorem season_games_count :
  total_teams = num_divisions * teams_per_division ∧
  (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * num_divisions +
  (teams_per_division * teams_per_division * inter_division_games) = total_games := by
  sorry

end season_games_count_l769_76958


namespace intersections_divisible_by_three_l769_76981

/-- The number of intersections between segments connecting points on parallel lines -/
def num_intersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n + 1) * n / 4

/-- Theorem stating that the number of intersections is divisible by 3 -/
theorem intersections_divisible_by_three (n : ℕ) :
  ∃ k : ℕ, num_intersections n = 3 * k :=
sorry

end intersections_divisible_by_three_l769_76981
