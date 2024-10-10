import Mathlib

namespace chips_ratio_l2110_211063

-- Define the total number of bags
def total_bags : ℕ := 3

-- Define the number of bags eaten for dinner
def dinner_bags : ℕ := 1

-- Define the number of bags eaten after dinner
def after_dinner_bags : ℕ := total_bags - dinner_bags

-- Theorem to prove
theorem chips_ratio :
  (after_dinner_bags : ℚ) / (dinner_bags : ℚ) = 2 / 1 :=
by sorry

end chips_ratio_l2110_211063


namespace probability_between_C_and_E_l2110_211088

/-- Given points A, B, C, D, E on a line segment AB, prove that the probability
    of a randomly selected point on AB being between C and E is 1/24. -/
theorem probability_between_C_and_E (A B C D E : ℝ) : 
  A < C ∧ C < E ∧ E < D ∧ D < B →  -- Points are ordered on the line
  B - A = 4 * (D - A) →            -- AB = 4AD
  B - A = 8 * (C - B) →            -- AB = 8BC
  B - E = 2 * (E - C) →            -- BE = 2CE
  (E - C) / (B - A) = 1 / 24 := by
    sorry

end probability_between_C_and_E_l2110_211088


namespace ax_plus_by_fifth_power_l2110_211056

theorem ax_plus_by_fifth_power (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 6)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = -360 := by sorry

end ax_plus_by_fifth_power_l2110_211056


namespace goldfish_cost_graph_is_finite_set_of_points_l2110_211086

def goldfish_cost (n : ℕ) : ℚ := 20 * n + 10

def valid_purchase (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∃ (S : Set (ℕ × ℚ)),
    Finite S ∧
    (∀ p ∈ S, valid_purchase p.1 ∧ p.2 = goldfish_cost p.1) ∧
    (∀ n, valid_purchase n → (n, goldfish_cost n) ∈ S) :=
  sorry

end goldfish_cost_graph_is_finite_set_of_points_l2110_211086


namespace q_sum_zero_five_l2110_211079

/-- A monic polynomial of degree 5 -/
def MonicPolynomial5 (q : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, q x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + q 0

/-- The main theorem -/
theorem q_sum_zero_five
  (q : ℝ → ℝ)
  (monic : MonicPolynomial5 q)
  (h1 : q 1 = 24)
  (h2 : q 2 = 48)
  (h3 : q 3 = 72) :
  q 0 + q 5 = 120 := by
  sorry

end q_sum_zero_five_l2110_211079


namespace largest_special_number_last_digit_l2110_211022

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_three_digits (n : ℕ) : ℕ := n / 10

theorem largest_special_number_last_digit :
  ∃ (n : ℕ), is_four_digit n ∧ 
             n % 9 = 0 ∧ 
             (first_three_digits n) % 4 = 0 ∧
             ∀ (m : ℕ), (is_four_digit m ∧ 
                         m % 9 = 0 ∧ 
                         (first_three_digits m) % 4 = 0) → 
                         m ≤ n ∧
             n % 10 = 3 :=
by sorry

end largest_special_number_last_digit_l2110_211022


namespace percentage_less_l2110_211087

theorem percentage_less (x y : ℝ) (h : x = 5 * y) : (x - y) / x * 100 = 80 :=
by
  sorry

end percentage_less_l2110_211087


namespace tens_digit_of_23_pow_2023_l2110_211030

theorem tens_digit_of_23_pow_2023 : ∃ n : ℕ, 23^2023 ≡ 60 + n [ZMOD 100] ∧ 0 ≤ n ∧ n < 10 :=
sorry

end tens_digit_of_23_pow_2023_l2110_211030


namespace sqrt_40000_l2110_211046

theorem sqrt_40000 : Real.sqrt 40000 = 200 := by
  sorry

end sqrt_40000_l2110_211046


namespace symmetry_properties_l2110_211089

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ

-- Define a quadratic function type
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to get the symmetric line about x-axis
def symmetricAboutXAxis (l : Line) : Line :=
  { a := -l.a, b := -l.b }

-- Function to get the symmetric line about y-axis
def symmetricAboutYAxis (l : Line) : Line :=
  { a := -l.a, b := l.b }

-- Function to get the symmetric quadratic function about origin
def symmetricAboutOrigin (q : QuadraticFunction) : QuadraticFunction :=
  { a := q.a, b := -q.b, c := -q.c }

-- Theorem statements
theorem symmetry_properties (l : Line) (q : QuadraticFunction) :
  (symmetricAboutXAxis l = { a := -l.a, b := -l.b }) ∧
  (symmetricAboutYAxis l = { a := -l.a, b := l.b }) ∧
  (symmetricAboutOrigin q = { a := q.a, b := -q.b, c := -q.c }) := by
  sorry


end symmetry_properties_l2110_211089


namespace bob_investment_l2110_211019

theorem bob_investment (fund_a_initial : ℝ) (fund_a_interest : ℝ) (fund_b_interest : ℝ) (difference : ℝ) :
  fund_a_initial = 2000 →
  fund_a_interest = 0.12 →
  fund_b_interest = 0.30 →
  difference = 549.9999999999998 →
  ∃ (fund_b_initial : ℝ),
    fund_a_initial * (1 + fund_a_interest) = 
    fund_b_initial * (1 + fund_b_interest)^2 + difference ∧
    fund_b_initial = 1000 :=
by sorry

end bob_investment_l2110_211019


namespace leading_coefficient_of_g_l2110_211078

/-- Given a polynomial g(x) that satisfies g(x + 1) - g(x) = 8x + 6 for all x,
    prove that the leading coefficient of g(x) is 4. -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) : 
  (∀ x, g (x + 1) - g x = 8 * x + 6) → 
  ∃ a b c : ℝ, (∀ x, g x = 4 * x^2 + a * x + b) ∧ c = 4 ∧ c * x^2 ≠ 0 := by
  sorry

end leading_coefficient_of_g_l2110_211078


namespace park_walk_distance_l2110_211023

theorem park_walk_distance (area : ℝ) (π_approx : ℝ) (extra_distance : ℝ) : 
  area = 616 →
  π_approx = 22 / 7 →
  extra_distance = 3 →
  2 * π_approx * (area / π_approx).sqrt + extra_distance = 91 :=
by
  sorry

end park_walk_distance_l2110_211023


namespace complement_intersection_equals_set_l2110_211053

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3}

-- Define set P
def P : Set ℕ := {1, 2}

-- Define set Q
def Q : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_equals_set : 
  (U \ (P ∩ Q)) = {1, 3} := by sorry

end complement_intersection_equals_set_l2110_211053


namespace painted_cube_probability_l2110_211090

/-- Represents a 3x3x3 cube with two adjacent faces painted -/
structure PaintedCube where
  size : Nat
  painted_faces : Nat

/-- Counts the number of cubes with exactly two painted faces -/
def count_two_painted (cube : PaintedCube) : Nat :=
  4  -- The edge cubes between the two painted faces

/-- Counts the number of cubes with no painted faces -/
def count_no_painted (cube : PaintedCube) : Nat :=
  9  -- The interior cubes not visible from any painted face

/-- Calculates the total number of ways to select two cubes -/
def total_selections (cube : PaintedCube) : Nat :=
  (cube.size^3 * (cube.size^3 - 1)) / 2

/-- The main theorem to prove -/
theorem painted_cube_probability (cube : PaintedCube) 
  (h1 : cube.size = 3) 
  (h2 : cube.painted_faces = 2) : 
  (count_two_painted cube * count_no_painted cube) / total_selections cube = 4 / 39 := by
  sorry


end painted_cube_probability_l2110_211090


namespace planes_perpendicular_to_line_are_parallel_parallel_planes_imply_line_parallel_l2110_211092

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relations
variable (perpendicular : P → L → Prop) -- Plane is perpendicular to a line
variable (parallel_planes : P → P → Prop) -- Two planes are parallel
variable (parallel_line_plane : L → P → Prop) -- A line is parallel to a plane
variable (line_in_plane : L → P → Prop) -- A line is in a plane

-- Theorem 1: If two planes are both perpendicular to the same line, then these two planes are parallel
theorem planes_perpendicular_to_line_are_parallel 
  (p1 p2 : P) (l : L) 
  (h1 : perpendicular p1 l) 
  (h2 : perpendicular p2 l) : 
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: If two planes are parallel to each other, then a line in one of the planes is parallel to the other plane
theorem parallel_planes_imply_line_parallel 
  (p1 p2 : P) (l : L) 
  (h1 : parallel_planes p1 p2) 
  (h2 : line_in_plane l p1) : 
  parallel_line_plane l p2 :=
sorry

end planes_perpendicular_to_line_are_parallel_parallel_planes_imply_line_parallel_l2110_211092


namespace board_tileable_iff_divisibility_l2110_211010

/-- A board is tileable if it can be covered completely with 3×1 tiles -/
def is_tileable (m n : ℕ) : Prop :=
  ∃ (tiling : Set (ℕ × ℕ × Bool)), 
    (∀ (tile : ℕ × ℕ × Bool), tile ∈ tiling → 
      (let (x, y, horizontal) := tile
       (x ≥ m ∨ y ≥ m) ∧ x < n ∧ y < n ∧
       (if horizontal then x + 2 < n else y + 2 < n))) ∧
    (∀ (i j : ℕ), m ≤ i ∧ i < n ∧ m ≤ j ∧ j < n → 
      ∃! (tile : ℕ × ℕ × Bool), tile ∈ tiling ∧
        (let (x, y, horizontal) := tile
         i ∈ Set.range (fun k => x + k) ∧ 
         j ∈ Set.range (fun k => y + k) ∧
         (if horizontal then x + 2 = i else y + 2 = j)))

/-- The main theorem -/
theorem board_tileable_iff_divisibility {m n : ℕ} (h_pos : 0 < m) (h_lt : m < n) :
  is_tileable m n ↔ (n - m) * (n + m) % 3 = 0 :=
sorry

end board_tileable_iff_divisibility_l2110_211010


namespace initial_cards_proof_l2110_211035

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := 455

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := 301

/-- The number of cards Nell has left -/
def cards_left : ℕ := 154

/-- Theorem stating that the initial number of cards is equal to the sum of cards given away and cards left -/
theorem initial_cards_proof : initial_cards = cards_given + cards_left := by
  sorry

end initial_cards_proof_l2110_211035


namespace total_cotton_yield_l2110_211016

/-- 
Given two cotton fields:
- Field 1 has m hectares and produces an average of a kilograms per hectare
- Field 2 has n hectares and produces an average of b kilograms per hectare
This theorem proves that the total cotton yield is am + bn kilograms
-/
theorem total_cotton_yield 
  (m n a b : ℝ) 
  (h1 : m ≥ 0) 
  (h2 : n ≥ 0) 
  (h3 : a ≥ 0) 
  (h4 : b ≥ 0) : 
  m * a + n * b = m * a + n * b := by
  sorry

end total_cotton_yield_l2110_211016


namespace fraction_simplification_l2110_211051

theorem fraction_simplification (a b c : ℝ) (h : 2*a - 3*c - 4 + b ≠ 0) :
  (6*a^2 - 2*b^2 + 6*c^2 + a*b - 13*a*c - 4*b*c - 18*a - 5*b + 17*c + 12) / 
  (4*a^2 - b^2 + 9*c^2 - 12*a*c - 16*a + 24*c + 16) = 
  (3*a - 2*c - 3 + 2*b) / (2*a - 3*c - 4 + b) := by
  sorry

end fraction_simplification_l2110_211051


namespace at_most_two_greater_than_one_l2110_211003

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬(2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) := by
  sorry

end at_most_two_greater_than_one_l2110_211003


namespace task_completion_probability_l2110_211091

theorem task_completion_probability (p1 p2 : ℚ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/5) : 
  p1 * (1 - p2) = 4/15 := by
sorry

end task_completion_probability_l2110_211091


namespace unique_number_with_conditions_l2110_211014

theorem unique_number_with_conditions : ∃! b : ℤ, 
  40 < b ∧ b < 120 ∧ 
  b % 4 = 3 ∧ 
  b % 5 = 3 ∧ 
  b % 6 = 3 ∧
  b = 63 := by sorry

end unique_number_with_conditions_l2110_211014


namespace one_eighth_divided_by_one_fourth_l2110_211074

theorem one_eighth_divided_by_one_fourth (a b c : ℚ) : 
  a = 1/8 → b = 1/4 → c = a / b → c = 1/2 := by sorry

end one_eighth_divided_by_one_fourth_l2110_211074


namespace balls_in_boxes_l2110_211076

/-- The number of ways to choose 2 boxes out of 4 -/
def choose_empty_boxes : ℕ := 6

/-- The number of ways to place 4 different balls into 2 boxes, with at least one ball in each box -/
def place_balls : ℕ := 14

/-- The total number of ways to place 4 different balls into 4 numbered boxes such that exactly two boxes are empty -/
def total_ways : ℕ := choose_empty_boxes * place_balls

theorem balls_in_boxes :
  total_ways = 84 :=
sorry

end balls_in_boxes_l2110_211076


namespace unique_quadratic_trinomial_l2110_211043

theorem unique_quadratic_trinomial : ∃! (a b c : ℝ), 
  (∀ x : ℝ, (a + 1) * x^2 + b * x + c = 0 → (∃! y : ℝ, y = x)) ∧
  (∀ x : ℝ, a * x^2 + (b + 1) * x + c = 0 → (∃! y : ℝ, y = x)) ∧
  (∀ x : ℝ, a * x^2 + b * x + (c + 1) = 0 → (∃! y : ℝ, y = x)) ∧
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 := by
  sorry

end unique_quadratic_trinomial_l2110_211043


namespace dimitri_burger_calories_l2110_211096

/-- Given that Dimitri eats 3 burgers per day and each burger has 20 calories,
    prove that the total calories consumed after two days is 120 calories. -/
theorem dimitri_burger_calories : 
  let burgers_per_day : ℕ := 3
  let calories_per_burger : ℕ := 20
  let days : ℕ := 2
  burgers_per_day * calories_per_burger * days = 120 := by
  sorry


end dimitri_burger_calories_l2110_211096


namespace smallest_number_l2110_211073

theorem smallest_number : 
  -5 < -Real.pi ∧ -5 < -Real.sqrt 3 ∧ -5 < 0 := by sorry

end smallest_number_l2110_211073


namespace relationship_abc_l2110_211034

theorem relationship_abc : 
  let a : ℝ := (0.3 : ℝ)^3
  let b : ℝ := 3^(0.3 : ℝ)
  let c : ℝ := (0.2 : ℝ)^3
  c < a ∧ a < b := by sorry

end relationship_abc_l2110_211034


namespace tan_equality_in_range_l2110_211069

theorem tan_equality_in_range (m : ℤ) :
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (1230 * π / 180) →
  m = -30 := by sorry

end tan_equality_in_range_l2110_211069


namespace log_equation_roots_l2110_211068

theorem log_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 0 ∧ x₁ + a > 0 ∧ x₁ + a ≠ 1 ∧
   x₂ > 0 ∧ x₂ + a > 0 ∧ x₂ + a ≠ 1 ∧
   Real.log (2 * x₁) / Real.log (x₁ + a) = 2 ∧
   Real.log (2 * x₂) / Real.log (x₂ + a) = 2) ↔
  (a > 0 ∧ a < 1/2) :=
sorry

end log_equation_roots_l2110_211068


namespace babysitter_hours_l2110_211062

/-- The number of hours Milly hires the babysitter -/
def hours : ℕ := sorry

/-- The hourly rate of the current babysitter -/
def current_rate : ℕ := 16

/-- The hourly rate of the new babysitter -/
def new_rate : ℕ := 12

/-- The extra charge per scream for the new babysitter -/
def scream_charge : ℕ := 3

/-- The number of times the kids scream per babysitting gig -/
def scream_count : ℕ := 2

/-- The amount saved by switching to the new babysitter -/
def savings : ℕ := 18

theorem babysitter_hours : 
  current_rate * hours = new_rate * hours + scream_charge * scream_count + savings :=
by sorry

end babysitter_hours_l2110_211062


namespace siblings_selection_probability_l2110_211050

theorem siblings_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_rina : ℚ)
  (h_ram : p_ram = 4/7)
  (h_ravi : p_ravi = 1/5)
  (h_rina : p_rina = 3/8) :
  p_ram * p_ravi * p_rina = 3/70 := by
sorry

end siblings_selection_probability_l2110_211050


namespace factorization_quadratic_l2110_211041

theorem factorization_quadratic (x : ℝ) : x^2 + 2*x = x*(x+2) := by
  sorry

end factorization_quadratic_l2110_211041


namespace min_value_of_quartic_plus_constant_l2110_211012

theorem min_value_of_quartic_plus_constant :
  ∃ (min : ℝ), min = 2023 ∧ ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ min := by
  sorry

end min_value_of_quartic_plus_constant_l2110_211012


namespace horse_and_saddle_cost_l2110_211059

/-- The total cost of a horse and saddle, given their relative costs -/
def total_cost (saddle_cost : ℕ) (horse_cost_multiplier : ℕ) : ℕ :=
  saddle_cost + horse_cost_multiplier * saddle_cost

/-- Theorem: The total cost of a horse and saddle is $5000 -/
theorem horse_and_saddle_cost :
  total_cost 1000 4 = 5000 := by
  sorry

end horse_and_saddle_cost_l2110_211059


namespace at_least_one_accepted_l2110_211095

theorem at_least_one_accepted (prob_A prob_B : ℝ) 
  (h1 : 0 ≤ prob_A ∧ prob_A ≤ 1)
  (h2 : 0 ≤ prob_B ∧ prob_B ≤ 1)
  (independence : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = prob_A + prob_B - prob_A * prob_B :=
by sorry

end at_least_one_accepted_l2110_211095


namespace girls_in_class_l2110_211007

/-- Proves the number of girls in a class with a given ratio and total students -/
theorem girls_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) :
  total = 63 ∧ girls_ratio = 4 ∧ boys_ratio = 3 →
  (girls_ratio * total) / (girls_ratio + boys_ratio) = 36 := by
  sorry

end girls_in_class_l2110_211007


namespace carpet_transformation_possible_l2110_211066

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a cut piece of a rectangle -/
structure CutPiece where
  width : ℕ
  height : ℕ

/-- Represents the state of the carpet after being cut -/
structure DamagedCarpet where
  original : Rectangle
  cutOut : CutPiece

/-- Function to check if a transformation from a damaged carpet to a new rectangle is possible -/
def canTransform (damaged : DamagedCarpet) (new : Rectangle) : Prop :=
  damaged.original.width * damaged.original.height - 
  damaged.cutOut.width * damaged.cutOut.height = 
  new.width * new.height

/-- The main theorem to prove -/
theorem carpet_transformation_possible : 
  ∃ (damaged : DamagedCarpet) (new : Rectangle),
    damaged.original = ⟨9, 12⟩ ∧ 
    damaged.cutOut = ⟨1, 8⟩ ∧
    new = ⟨10, 10⟩ ∧
    canTransform damaged new :=
sorry

end carpet_transformation_possible_l2110_211066


namespace scrap_rate_cost_relationship_l2110_211029

/-- Represents the regression line equation for pig iron cost -/
def regression_line (x : ℝ) : ℝ := 256 + 2 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_relationship :
  ∀ x : ℝ, regression_line (x + 1) = regression_line x + 2 :=
by sorry

end scrap_rate_cost_relationship_l2110_211029


namespace quadratic_function_properties_l2110_211055

/-- A quadratic function f with parameter t -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (x - (t + 2) / 2)^2 - t^2 / 4

/-- The theorem stating the properties of the quadratic function and the value of t -/
theorem quadratic_function_properties (t : ℝ) :
  t ≠ 0 ∧
  f t ((t + 2) / 2) = -t^2 / 4 ∧
  f t 1 = 0 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2), f t x ≥ -5) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) (1/2), f t x = -5) →
  t = -9/2 :=
by sorry

end quadratic_function_properties_l2110_211055


namespace nuts_division_l2110_211099

theorem nuts_division (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) 
  (h1 : bags = 65) 
  (h2 : nuts_per_bag = 15) 
  (h3 : students = 13) : 
  (bags * nuts_per_bag) / students = 75 := by
  sorry

#check nuts_division

end nuts_division_l2110_211099


namespace cubic_factorization_l2110_211038

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end cubic_factorization_l2110_211038


namespace a_less_than_11_necessary_not_sufficient_l2110_211011

theorem a_less_than_11_necessary_not_sufficient :
  (∀ a : ℝ, (∃ x : ℝ, x^2 - 2*x + a < 0) → a < 11) ∧
  (∃ a : ℝ, a < 11 ∧ ¬(∃ x : ℝ, x^2 - 2*x + a < 0)) :=
sorry

end a_less_than_11_necessary_not_sufficient_l2110_211011


namespace fourth_root_13824000_l2110_211054

theorem fourth_root_13824000 : (62 : ℕ)^4 = 13824000 := by
  sorry

end fourth_root_13824000_l2110_211054


namespace no_consecutive_sum_for_2_14_l2110_211067

theorem no_consecutive_sum_for_2_14 : ¬∃ (k n : ℕ), k > 0 ∧ n > 0 ∧ 2^14 = (k * (2*n + k + 1)) / 2 := by
  sorry

end no_consecutive_sum_for_2_14_l2110_211067


namespace equation_solutions_l2110_211039

theorem equation_solutions (x : ℝ) : 
  (x^3 + 2*x)^(1/5) = (x^5 - 2*x)^(1/3) ↔ x = 0 ∨ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
sorry

end equation_solutions_l2110_211039


namespace max_sum_of_squares_l2110_211036

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18)
  (h2 : a * b + c + d = 85)
  (h3 : a * d + b * c = 187)
  (h4 : c * d = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 18 ∧ 
    a' * b' + c' + d' = 85 ∧ 
    a' * d' + b' * c' = 187 ∧ 
    c' * d' = 110 ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 = 120 :=
by sorry

#check max_sum_of_squares

end max_sum_of_squares_l2110_211036


namespace cube_roots_opposite_implies_a_eq_neg_three_l2110_211044

theorem cube_roots_opposite_implies_a_eq_neg_three (a : ℝ) :
  (∃ x : ℝ, x^3 = 2*a + 1 ∧ (-x)^3 = 2 - a) → a = -3 := by
sorry

end cube_roots_opposite_implies_a_eq_neg_three_l2110_211044


namespace quadratic_equation_from_sum_and_difference_l2110_211075

theorem quadratic_equation_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_abs : |x - y| = 12) : 
  (∀ z : ℝ, z^2 - 10*z - 11 = 0 ↔ z = x ∨ z = y) := by
  sorry

end quadratic_equation_from_sum_and_difference_l2110_211075


namespace investment_change_l2110_211083

/-- Proves that an investment of $200 with a 20% loss followed by a 25% gain results in 0% change --/
theorem investment_change (initial_investment : ℝ) (first_year_loss_percent : ℝ) (second_year_gain_percent : ℝ) :
  initial_investment = 200 →
  first_year_loss_percent = 20 →
  second_year_gain_percent = 25 →
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let final_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  final_amount = initial_investment := by
  sorry

#check investment_change

end investment_change_l2110_211083


namespace library_books_theorem_l2110_211049

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new edition
variable (is_new_edition : Book → Prop)

-- Theorem stating that if not all books are new editions,
-- then there exists a book that is not a new edition and not all books are new editions
theorem library_books_theorem (h : ¬ ∀ b : Book, is_new_edition b) :
  (∃ b : Book, ¬ is_new_edition b) ∧ (¬ ∀ b : Book, is_new_edition b) :=
by sorry

end library_books_theorem_l2110_211049


namespace inscribed_quadrilateral_radius_l2110_211042

/-- Given a quadrilateral ABCD inscribed in a circle with diagonals intersecting at M,
    where AB = a, CD = b, and ∠AMB = α, the radius R of the circle is as follows. -/
theorem inscribed_quadrilateral_radius 
  (a b : ℝ) (α : ℝ) (ha : a > 0) (hb : b > 0) (hα : 0 < α ∧ α < π) :
  ∃ (R : ℝ), R = (Real.sqrt (a^2 + b^2 + 2*a*b*(Real.cos α))) / (2 * Real.sin α) :=
sorry

end inscribed_quadrilateral_radius_l2110_211042


namespace rectangle_perimeter_l2110_211082

theorem rectangle_perimeter (length width diagonal : ℝ) : 
  length = 8 ∧ diagonal = 17 ∧ length^2 + width^2 = diagonal^2 →
  2 * (length + width) = 46 := by
sorry

end rectangle_perimeter_l2110_211082


namespace red_peaches_count_l2110_211070

/-- The number of red peaches in a basket with yellow, green, and red peaches. -/
def num_red_peaches (yellow green red_and_green : ℕ) : ℕ :=
  red_and_green - green

/-- Theorem stating that the number of red peaches is 6. -/
theorem red_peaches_count :
  let yellow : ℕ := 90
  let green : ℕ := 16
  let red_and_green : ℕ := 22
  num_red_peaches yellow green red_and_green = 6 := by
  sorry

end red_peaches_count_l2110_211070


namespace intersection_A_B_l2110_211025

def set_A : Set ℝ := {x | x^2 - 2*x < 0}
def set_B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_A_B : set_A ∩ set_B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_A_B_l2110_211025


namespace range_of_x_l2110_211001

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) : f (x^2 - 4) < 2 → x ∈ Set.Ioo (-Real.sqrt 5) (-2) ∪ Set.Ioo 2 (Real.sqrt 5) := by
  sorry

end range_of_x_l2110_211001


namespace simplify_sqrt_product_l2110_211004

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^4 * 3^5) = 675 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_product_l2110_211004


namespace number_of_divisors_of_30_l2110_211002

theorem number_of_divisors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end number_of_divisors_of_30_l2110_211002


namespace emmy_has_200_l2110_211006

/-- The amount of money Emmy has -/
def emmys_money : ℕ := sorry

/-- The amount of money Gerry has -/
def gerrys_money : ℕ := 100

/-- The cost of one apple -/
def apple_cost : ℕ := 2

/-- The total number of apples Emmy and Gerry can buy -/
def total_apples : ℕ := 150

/-- Theorem: Emmy has $200 -/
theorem emmy_has_200 : emmys_money = 200 := by
  have total_cost : ℕ := apple_cost * total_apples
  have sum_of_money : emmys_money + gerrys_money = total_cost := sorry
  sorry

end emmy_has_200_l2110_211006


namespace brendas_age_l2110_211028

/-- Proves that Brenda's age is 8/3 years given the conditions in the problem. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)   -- Addison's age is four times Brenda's age
  (h2 : J = B + 8)   -- Janet is eight years older than Brenda
  (h3 : A = J)       -- Addison and Janet are twins
  : B = 8 / 3 := by
  sorry

end brendas_age_l2110_211028


namespace problem_1_l2110_211048

theorem problem_1 (x y z : ℝ) : -x * y^2 * z^3 * (-x^2 * y)^3 = x^7 * y^5 * z^3 := by
  sorry

end problem_1_l2110_211048


namespace beach_house_pool_problem_l2110_211093

theorem beach_house_pool_problem (total_people : ℕ) (legs_in_pool : ℕ) (legs_per_person : ℕ) :
  total_people = 14 →
  legs_in_pool = 16 →
  legs_per_person = 2 →
  total_people - (legs_in_pool / legs_per_person) = 6 :=
by
  sorry

end beach_house_pool_problem_l2110_211093


namespace special_geometric_sequence_ratio_l2110_211032

/-- A geometric sequence with a special property -/
def SpecialGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) ∧ a 1 + a 5 = a 1 * a 5

/-- The ratio of the 13th to the 9th term is 9 -/
theorem special_geometric_sequence_ratio
  (a : ℕ → ℝ) (h : SpecialGeometricSequence a) :
  a 13 / a 9 = 9 := by
  sorry

end special_geometric_sequence_ratio_l2110_211032


namespace trapezoid_area_l2110_211094

/-- Represents a trapezoid ABCD with a circle passing through A, B, and touching C -/
structure TrapezoidWithCircle where
  /-- Length of CD -/
  cd : ℝ
  /-- Length of AE -/
  ae : ℝ
  /-- The circle is centered on diagonal AC -/
  circle_on_diagonal : Bool
  /-- BC is parallel to AD -/
  bc_parallel_ad : Bool
  /-- The circle passes through A and B -/
  circle_through_ab : Bool
  /-- The circle touches CD at C -/
  circle_touches_cd : Bool
  /-- The circle intersects AD at E -/
  circle_intersects_ad : Bool

/-- Calculate the area of the trapezoid ABCD -/
def calculate_area (t : TrapezoidWithCircle) : ℝ :=
  sorry

/-- Theorem stating that the area of the trapezoid ABCD is 204 -/
theorem trapezoid_area (t : TrapezoidWithCircle) 
  (h1 : t.cd = 6 * Real.sqrt 13)
  (h2 : t.ae = 8)
  (h3 : t.circle_on_diagonal)
  (h4 : t.bc_parallel_ad)
  (h5 : t.circle_through_ab)
  (h6 : t.circle_touches_cd)
  (h7 : t.circle_intersects_ad) :
  calculate_area t = 204 :=
sorry

end trapezoid_area_l2110_211094


namespace brave_children_count_l2110_211058

/-- Represents the arrangement of children on a bench -/
structure BenchArrangement where
  total_children : ℕ
  boy_girl_pairs : ℕ

/-- The initial arrangement with 2 children -/
def initial_arrangement : BenchArrangement :=
  { total_children := 2, boy_girl_pairs := 1 }

/-- The final arrangement with 22 children alternating boy-girl -/
def final_arrangement : BenchArrangement :=
  { total_children := 22, boy_girl_pairs := 21 }

/-- A child is brave if they create two new boy-girl pairs when sitting down -/
def brave_children (initial final : BenchArrangement) : ℕ :=
  (final.boy_girl_pairs - initial.boy_girl_pairs) / 2

theorem brave_children_count :
  brave_children initial_arrangement final_arrangement = 10 := by
  sorry

end brave_children_count_l2110_211058


namespace polynomial_root_implies_coefficients_l2110_211057

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*I : ℂ) ^ 3 + a * (2 - 3*I : ℂ) ^ 2 - (2 - 3*I : ℂ) + b = 0) : 
  a = -1/2 ∧ b = 91/2 := by
sorry

end polynomial_root_implies_coefficients_l2110_211057


namespace two_digit_numbers_count_l2110_211000

def digits_a : Finset Nat := {1, 2, 3, 4, 5, 6}
def digits_b : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n ≤ 99

def count_two_digit_numbers (digits : Finset Nat) : Nat :=
  (digits.filter (λ d => d > 0)).card * digits.card

theorem two_digit_numbers_count :
  (count_two_digit_numbers digits_a = 36) ∧
  (count_two_digit_numbers digits_b = 42) := by sorry

end two_digit_numbers_count_l2110_211000


namespace even_sum_probability_l2110_211013

/-- Represents a 4x4 grid filled with numbers from 1 to 16 -/
def Grid := Fin 4 → Fin 4 → Fin 16

/-- Checks if a list of numbers has an even sum -/
def hasEvenSum (l : List (Fin 16)) : Prop :=
  (l.map (fun x => x.val + 1)).sum % 2 = 0

/-- Checks if all rows and columns in a grid have even sums -/
def allRowsAndColumnsEven (g : Grid) : Prop :=
  (∀ i : Fin 4, hasEvenSum [g i 0, g i 1, g i 2, g i 3]) ∧
  (∀ j : Fin 4, hasEvenSum [g 0 j, g 1 j, g 2 j, g 3 j])

/-- The total number of ways to arrange 16 numbers in a 4x4 grid -/
def totalArrangements : ℕ := 20922789888000

/-- The number of valid arrangements with even sums in all rows and columns -/
def validArrangements : ℕ := 36

theorem even_sum_probability :
  (validArrangements : ℚ) / totalArrangements =
  (36 : ℚ) / 20922789888000 :=
sorry

end even_sum_probability_l2110_211013


namespace pie_crust_flour_usage_l2110_211018

/-- Given that 30 pie crusts each use 1/6 cup of flour, and 25 new pie crusts use
    the same total amount of flour, prove that each new pie crust uses 1/5 cup of flour. -/
theorem pie_crust_flour_usage
  (original_crusts : ℕ)
  (original_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (h1 : original_crusts = 30)
  (h2 : original_flour_per_crust = 1/6)
  (h3 : new_crusts = 25)
  (h4 : original_crusts * original_flour_per_crust = new_crusts * new_flour_per_crust) :
  new_flour_per_crust = 1/5 :=
sorry

end pie_crust_flour_usage_l2110_211018


namespace min_value_of_max_function_l2110_211052

theorem min_value_of_max_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ∃ (t : ℝ), t = 4 ∧ ∀ (s : ℝ), s ≥ max (x^2) (4 / (y * (x - y))) → s ≥ t :=
sorry

end min_value_of_max_function_l2110_211052


namespace city_connections_l2110_211037

/-- The number of cities in the problem -/
def num_cities : ℕ := 6

/-- The function to calculate the number of unique pairwise connections -/
def unique_connections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 6 cities, the number of unique pairwise connections is 15 -/
theorem city_connections : unique_connections num_cities = 15 := by
  sorry

end city_connections_l2110_211037


namespace tan_squared_sum_l2110_211026

theorem tan_squared_sum (x y : ℝ) 
  (h : 2 * Real.sin x * Real.sin y + 3 * Real.cos y + 6 * Real.cos x * Real.sin y = 7) : 
  Real.tan x ^ 2 + 2 * Real.tan y ^ 2 = 9 := by
  sorry

end tan_squared_sum_l2110_211026


namespace bottles_used_first_game_l2110_211064

theorem bottles_used_first_game 
  (total_bottles : ℕ)
  (bottles_left : ℕ)
  (bottles_used_second : ℕ)
  (h1 : total_bottles = 200)
  (h2 : bottles_left = 20)
  (h3 : bottles_used_second = 110) :
  total_bottles - bottles_left - bottles_used_second = 70 :=
by sorry

end bottles_used_first_game_l2110_211064


namespace correct_algebraic_operation_l2110_211009

theorem correct_algebraic_operation (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := by
  sorry

end correct_algebraic_operation_l2110_211009


namespace prop_one_correct_prop_two_not_always_true_prop_three_not_always_true_l2110_211031

-- Define the custom distance function
def customDist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

-- Proposition 1
theorem prop_one_correct (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) 
  (h₁ : x₀ ∈ Set.Icc x₁ x₂) (h₂ : y₀ ∈ Set.Icc y₁ y₂) :
  customDist x₁ y₁ x₀ y₀ + customDist x₀ y₀ x₂ y₂ = customDist x₁ y₁ x₂ y₂ := by sorry

-- Proposition 2
theorem prop_two_not_always_true :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    customDist x₁ y₁ x₂ y₂ + customDist x₂ y₂ x₃ y₃ ≤ customDist x₁ y₁ x₃ y₃ := by sorry

-- Proposition 3
theorem prop_three_not_always_true :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∧ 
    (customDist x₁ y₁ x₂ y₂)^2 + (customDist x₁ y₁ x₃ y₃)^2 ≠ (customDist x₂ y₂ x₃ y₃)^2 := by sorry

end prop_one_correct_prop_two_not_always_true_prop_three_not_always_true_l2110_211031


namespace average_age_when_youngest_born_l2110_211015

theorem average_age_when_youngest_born (total_people : ℕ) (current_average_age : ℝ) (youngest_age : ℝ) :
  total_people = 7 →
  current_average_age = 30 →
  youngest_age = 7 →
  (total_people * current_average_age - (total_people - 1) * youngest_age) / (total_people - 1) = 28 := by
  sorry

end average_age_when_youngest_born_l2110_211015


namespace building_height_proof_l2110_211005

/-- Proves the height of the first 10 stories in a 20-story building -/
theorem building_height_proof (total_stories : Nat) (first_section : Nat) (height_difference : Nat) (total_height : Nat) :
  total_stories = 20 →
  first_section = 10 →
  height_difference = 3 →
  total_height = 270 →
  ∃ (first_story_height : Nat),
    first_story_height * first_section + (first_story_height + height_difference) * (total_stories - first_section) = total_height ∧
    first_story_height = 12 := by
  sorry

end building_height_proof_l2110_211005


namespace acme_cheaper_at_min_shirts_l2110_211033

/-- Acme's setup fee -/
def acme_setup : ℕ := 75

/-- Acme's per-shirt cost -/
def acme_per_shirt : ℕ := 8

/-- Gamma's per-shirt cost -/
def gamma_per_shirt : ℕ := 16

/-- The minimum number of shirts for which Acme becomes cheaper than Gamma -/
def min_shirts : ℕ := 10

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts < gamma_per_shirt * min_shirts ∧
  ∀ n : ℕ, n < min_shirts →
    acme_setup + acme_per_shirt * n ≥ gamma_per_shirt * n :=
by sorry

end acme_cheaper_at_min_shirts_l2110_211033


namespace vector_dot_product_equals_three_l2110_211071

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  ab_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1
  bc_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem vector_dot_product_equals_three 
  (A B C M : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hm : vec_sub B M = vec_scale 2 (vec_sub A M)) : 
  dot_product (vec_sub C M) (vec_sub C A) = 3 := by
  sorry


end vector_dot_product_equals_three_l2110_211071


namespace unique_right_triangle_area_twice_perimeter_l2110_211061

/-- A right triangle with integer leg lengths -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  hyp : c^2 = a^2 + b^2  -- Pythagorean theorem

/-- The area of a right triangle is equal to twice its perimeter -/
def areaEqualsTwicePerimeter (t : RightTriangle) : Prop :=
  (t.a * t.b : ℕ) = 4 * (t.a + t.b + t.c)

/-- There exists exactly one right triangle with integer leg lengths
    where the area is equal to twice the perimeter -/
theorem unique_right_triangle_area_twice_perimeter :
  ∃! t : RightTriangle, areaEqualsTwicePerimeter t := by sorry

end unique_right_triangle_area_twice_perimeter_l2110_211061


namespace product_one_inequality_l2110_211060

theorem product_one_inequality (a b c d e : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_prod : a * b * c * d * e = 1) : 
  a^2 / b^2 + b^2 / c^2 + c^2 / d^2 + d^2 / e^2 + e^2 / a^2 ≥ a + b + c + d + e := by
sorry

end product_one_inequality_l2110_211060


namespace rectangle_max_area_l2110_211040

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 40 →
  l * w ≤ 100 :=
by sorry

end rectangle_max_area_l2110_211040


namespace target_word_satisfies_conditions_target_word_is_unique_l2110_211097

/-- Represents a word with multiple meanings -/
structure MultiMeaningWord where
  word : String
  soundsLike : String
  usedInSports : Bool
  usedInPensions : Bool

/-- Represents the conditions for the word we're looking for -/
def wordConditions : MultiMeaningWord → Prop := fun w =>
  w.soundsLike = "festive dance event" ∧
  w.usedInSports = true ∧
  w.usedInPensions = true

/-- The word we're looking for -/
def targetWord : MultiMeaningWord := {
  word := "баллы",
  soundsLike := "festive dance event",
  usedInSports := true,
  usedInPensions := true
}

/-- Theorem stating that our target word satisfies all conditions -/
theorem target_word_satisfies_conditions : 
  wordConditions targetWord := by sorry

/-- Theorem stating that our target word is unique -/
theorem target_word_is_unique :
  ∀ w : MultiMeaningWord, wordConditions w → w = targetWord := by sorry

end target_word_satisfies_conditions_target_word_is_unique_l2110_211097


namespace sequence_general_term_l2110_211081

/-- Given a sequence {a_n} with sum of first n terms S_n = (2/3)a_n + 1/3,
    prove that the general term formula is a_n = (-2)^(n-1) -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (2/3) * a n + 1/3) →
  ∃ C : ℝ, ∀ n : ℕ, a n = C * (-2)^(n-1) :=
sorry

end sequence_general_term_l2110_211081


namespace inequalities_proof_l2110_211017

theorem inequalities_proof (a b c : ℝ) (ha : a > 0) (hbc : a < b ∧ b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a + b < b + c) ∧ 
  (c / a > 1) := by
  sorry

end inequalities_proof_l2110_211017


namespace cone_max_volume_surface_ratio_l2110_211024

/-- For a cone with slant height 2, the ratio of its volume to its lateral surface area
    is maximized when the radius of its base is √2. -/
theorem cone_max_volume_surface_ratio (r : ℝ) (h : ℝ) : 
  let l : ℝ := 2
  let S := 2 * Real.pi * r
  let V := (1/3) * Real.pi * r^2 * Real.sqrt (l^2 - r^2)
  (∀ r' : ℝ, 0 < r' → V / S ≤ ((1/3) * Real.pi * r'^2 * Real.sqrt (l^2 - r'^2)) / (2 * Real.pi * r')) →
  r = Real.sqrt 2 :=
sorry

end cone_max_volume_surface_ratio_l2110_211024


namespace equation_solution_l2110_211020

theorem equation_solution (x : ℝ) : 
  Real.sqrt (1 + Real.sqrt (4 + Real.sqrt (2 * x + 3))) = (1 + Real.sqrt (2 * x + 3)) ^ (1/4) → 
  x = -23/32 := by
sorry

end equation_solution_l2110_211020


namespace digit_count_8_pow_12_times_5_pow_18_l2110_211085

theorem digit_count_8_pow_12_times_5_pow_18 : 
  (Nat.log 10 (8^12 * 5^18) + 1 : ℕ) = 24 := by sorry

end digit_count_8_pow_12_times_5_pow_18_l2110_211085


namespace compound_molecular_weight_l2110_211065

/-- Atomic weight in atomic mass units (amu) -/
def atomic_weight (element : String) : Float :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | "O" => 16.00
  | "C" => 12.01
  | _ => 0  -- Default case for unknown elements

/-- Number of atoms for each element in the compound -/
def atom_count (element : String) : Nat :=
  match element with
  | "N" => 2
  | "H" => 6
  | "Br" => 1
  | "O" => 1
  | "C" => 3
  | _ => 0  -- Default case for elements not in the compound

/-- Calculate the molecular weight of the compound -/
def molecular_weight : Float :=
  ["N", "H", "Br", "O", "C"].map (fun e => (atomic_weight e) * (atom_count e).toFloat)
    |> List.sum

/-- Theorem: The molecular weight of the compound is approximately 166.01 amu -/
theorem compound_molecular_weight :
  (molecular_weight - 166.01).abs < 0.01 := by
  sorry

end compound_molecular_weight_l2110_211065


namespace bus_dispatch_interval_l2110_211027

/-- The speed of the bus -/
def bus_speed : ℝ := sorry

/-- The speed of Xiao Wang -/
def person_speed : ℝ := sorry

/-- The interval between each bus dispatch in minutes -/
def dispatch_interval : ℝ := sorry

/-- The time between a bus passing Xiao Wang from behind in minutes -/
def overtake_time : ℝ := 6

/-- The time between a bus coming towards Xiao Wang in minutes -/
def approach_time : ℝ := 3

/-- Theorem stating that given the conditions, the dispatch interval is 4 minutes -/
theorem bus_dispatch_interval : 
  bus_speed > 0 ∧ 
  person_speed > 0 ∧ 
  person_speed < bus_speed ∧
  overtake_time * (bus_speed - person_speed) = dispatch_interval * bus_speed ∧
  approach_time * (bus_speed + person_speed) = dispatch_interval * bus_speed →
  dispatch_interval = 4 := by sorry

end bus_dispatch_interval_l2110_211027


namespace prob_at_least_one_l2110_211072

/-- The probability of possessing at least one of two independent events,
    given their individual probabilities -/
theorem prob_at_least_one (p_ballpoint p_ink : ℚ) 
  (h_ballpoint : p_ballpoint = 3/5)
  (h_ink : p_ink = 2/3)
  (h_independent : True) -- Assumption of independence
  : p_ballpoint + p_ink - p_ballpoint * p_ink = 13/15 := by
  sorry

end prob_at_least_one_l2110_211072


namespace complement_intersection_theorem_l2110_211077

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2} := by sorry

end complement_intersection_theorem_l2110_211077


namespace min_questions_correct_l2110_211047

/-- Represents a company with N people, where one person knows everyone but is known by no one. -/
structure Company (N : ℕ) where
  -- The number of people in the company is at least 2
  people_count : N ≥ 2
  -- The function that determines if person i knows person j
  knows : Fin N → Fin N → Bool
  -- There exists a person who knows everyone else but is known by no one
  exists_z : ∃ z : Fin N, (∀ i : Fin N, i ≠ z → knows z i) ∧ (∀ i : Fin N, i ≠ z → ¬knows i z)

/-- The minimum number of questions needed to identify the person Z -/
def min_questions (N : ℕ) (c : Company N) : ℕ := N - 1

/-- Theorem stating that the minimum number of questions needed is N - 1 -/
theorem min_questions_correct (N : ℕ) (c : Company N) :
  ∀ strategy : (Fin N → Fin N → Bool) → Fin N,
  (∀ knows : Fin N → Fin N → Bool, 
   ∃ z : Fin N, (∀ i : Fin N, i ≠ z → knows z i) ∧ (∀ i : Fin N, i ≠ z → ¬knows i z) →
   ∃ questions : Finset (Fin N × Fin N),
     questions.card ≥ min_questions N c ∧
     strategy knows = z) :=
by
  sorry

end min_questions_correct_l2110_211047


namespace mary_anne_sparkling_water_cost_l2110_211008

/-- The amount Mary Anne spends on sparkling water in a year -/
def sparkling_water_cost (daily_consumption : ℚ) (bottle_cost : ℚ) : ℚ :=
  (365 : ℚ) * daily_consumption * bottle_cost

theorem mary_anne_sparkling_water_cost :
  sparkling_water_cost (1/5) 2 = 146 := by
  sorry

end mary_anne_sparkling_water_cost_l2110_211008


namespace arithmetic_sequence_ninth_term_l2110_211084

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 14 = 2) :
  a 9 = 1 := by
sorry

end arithmetic_sequence_ninth_term_l2110_211084


namespace books_ratio_l2110_211021

/-- Given the number of books Elmo, Laura, and Stu have, prove the ratio of Laura's books to Stu's books -/
theorem books_ratio (elmo_books laura_books stu_books : ℕ) : 
  elmo_books = 24 →
  stu_books = 4 →
  elmo_books = 3 * laura_books →
  laura_books / stu_books = 2 := by
sorry

end books_ratio_l2110_211021


namespace total_fertilizer_used_l2110_211045

/-- The amount of fertilizer used per day for the first 9 days -/
def normal_amount : ℕ := 2

/-- The number of days the florist uses the normal amount of fertilizer -/
def normal_days : ℕ := 9

/-- The extra amount of fertilizer used on the final day -/
def extra_amount : ℕ := 4

/-- The total number of days the florist uses fertilizer -/
def total_days : ℕ := normal_days + 1

/-- Theorem: The total amount of fertilizer used over 10 days is 24 pounds -/
theorem total_fertilizer_used : 
  normal_amount * normal_days + (normal_amount + extra_amount) = 24 := by
  sorry

end total_fertilizer_used_l2110_211045


namespace min_removals_for_three_by_three_l2110_211080

/-- Represents a 3x3 square figure made of matches -/
structure MatchSquare where
  size : Nat
  total_matches : Nat
  matches_per_side : Nat

/-- Defines the properties of our specific 3x3 match square -/
def three_by_three_square : MatchSquare :=
  { size := 3
  , total_matches := 24
  , matches_per_side := 1 }

/-- Defines what it means for a number of removals to be valid -/
def is_valid_removal (square : MatchSquare) (removals : Nat) : Prop :=
  removals ≤ square.total_matches ∧
  ∀ (x y : Nat), x < square.size ∧ y < square.size →
    ∃ (side : Nat), side < 4 ∧ 
      (removals > (x * square.size + y) * 4 + side)

/-- The main theorem statement -/
theorem min_removals_for_three_by_three (square : MatchSquare) 
  (h1 : square = three_by_three_square) :
  ∃ (n : Nat), is_valid_removal square n ∧
    ∀ (m : Nat), m < n → ¬ is_valid_removal square m :=
  sorry

end min_removals_for_three_by_three_l2110_211080


namespace book_selling_price_total_selling_price_is_595_l2110_211098

/-- Calculates the total selling price of two books given the following conditions:
    - Total cost of two books is 600
    - First book is sold at a loss of 15%
    - Second book is sold at a gain of 19%
    - Cost of the book sold at a loss is 350
-/
theorem book_selling_price (total_cost : ℝ) (loss_percentage : ℝ) (gain_percentage : ℝ) (loss_book_cost : ℝ) : ℝ :=
  let selling_price_loss_book := loss_book_cost * (1 - loss_percentage / 100)
  let gain_book_cost := total_cost - loss_book_cost
  let selling_price_gain_book := gain_book_cost * (1 + gain_percentage / 100)
  selling_price_loss_book + selling_price_gain_book

theorem total_selling_price_is_595 :
  book_selling_price 600 15 19 350 = 595 := by
  sorry

end book_selling_price_total_selling_price_is_595_l2110_211098
