import Mathlib

namespace complex_number_equality_l1743_174346

theorem complex_number_equality : (((1 + Complex.I)^4) / (1 - Complex.I)) + 2 = -2 * Complex.I := by
  sorry

end complex_number_equality_l1743_174346


namespace bagel_cut_theorem_l1743_174385

/-- Number of pieces resulting from cutting a torus-shaped object -/
def torusPieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: Cutting a torus-shaped object (bagel) with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem :
  torusPieces 10 = 11 := by
  sorry

end bagel_cut_theorem_l1743_174385


namespace stream_speed_l1743_174330

theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 78)
  (h2 : upstream_distance = 50)
  (h3 : time = 2) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 7 := by
  sorry

end stream_speed_l1743_174330


namespace B_power_101_l1743_174323

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 :
  B ^ 101 = ![![0, 0, 1],
              ![1, 0, 0],
              ![0, 1, 0]] := by
  sorry

end B_power_101_l1743_174323


namespace inequality_equivalence_l1743_174364

theorem inequality_equivalence (x : ℝ) : 
  (|x + 3| + |1 - x|) / (x + 2016) < 1 ↔ x < -2016 ∨ (-1009 < x ∧ x < 1007) :=
by sorry

end inequality_equivalence_l1743_174364


namespace movie_production_people_l1743_174398

/-- The number of people at the movie production --/
def num_people : ℕ := 50

/-- The cost of hiring actors --/
def actor_cost : ℕ := 1200

/-- The cost of food per person --/
def food_cost_per_person : ℕ := 3

/-- The total cost of the movie production --/
def total_cost : ℕ := 10000 - 5950

/-- The equipment rental cost is twice the combined cost of food and actors --/
def equipment_cost (p : ℕ) : ℕ := 2 * (food_cost_per_person * p + actor_cost)

/-- The total cost calculation based on the number of people --/
def calculated_cost (p : ℕ) : ℕ :=
  actor_cost + food_cost_per_person * p + equipment_cost p

theorem movie_production_people :
  calculated_cost num_people = total_cost :=
by sorry

end movie_production_people_l1743_174398


namespace total_eggs_proof_l1743_174339

/-- The total number of eggs used by Molly's employees at the Wafting Pie Company -/
def total_eggs (morning_eggs afternoon_eggs : ℕ) : ℕ :=
  morning_eggs + afternoon_eggs

/-- Proof that the total number of eggs used is 1339 -/
theorem total_eggs_proof (morning_eggs afternoon_eggs : ℕ) 
  (h1 : morning_eggs = 816) 
  (h2 : afternoon_eggs = 523) : 
  total_eggs morning_eggs afternoon_eggs = 1339 := by
  sorry

#eval total_eggs 816 523

end total_eggs_proof_l1743_174339


namespace omega_on_real_axis_l1743_174326

theorem omega_on_real_axis (z : ℂ) (h1 : z.re ≠ 0) (h2 : Complex.abs z = 1) :
  let ω := z + z⁻¹
  ω.im = 0 := by
  sorry

end omega_on_real_axis_l1743_174326


namespace smallest_solutions_l1743_174314

/-- The function that checks if a given positive integer k satisfies the equation cos²(k² + 6²)° = 1 --/
def satisfies_equation (k : ℕ+) : Prop :=
  (Real.cos ((k.val ^ 2 + 6 ^ 2 : ℕ) : ℝ) * Real.pi / 180) ^ 2 = 1

/-- Theorem stating that 12 and 18 are the two smallest positive integers satisfying the equation --/
theorem smallest_solutions : 
  (satisfies_equation 12) ∧ 
  (satisfies_equation 18) ∧ 
  (∀ k : ℕ+, k < 12 → ¬(satisfies_equation k)) ∧
  (∀ k : ℕ+, 12 < k → k < 18 → ¬(satisfies_equation k)) :=
sorry

end smallest_solutions_l1743_174314


namespace no_integer_solution_for_equation_l1743_174377

theorem no_integer_solution_for_equation :
  ∀ x y : ℤ, x^2 - 3*y^2 ≠ 17 := by
  sorry

end no_integer_solution_for_equation_l1743_174377


namespace complement_of_intersection_main_theorem_l1743_174356

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem statement
theorem complement_of_intersection (x : Nat) : 
  x ∈ (A ∩ B)ᶜ ↔ (x ∈ U ∧ x ∉ (A ∩ B)) :=
by
  sorry

-- Main theorem to prove
theorem main_theorem : (A ∩ B)ᶜ = {1, 4} :=
by
  sorry

end complement_of_intersection_main_theorem_l1743_174356


namespace cube_edge_sum_l1743_174311

theorem cube_edge_sum (surface_area : ℝ) (h : surface_area = 150) :
  let side_length := Real.sqrt (surface_area / 6)
  12 * side_length = 60 := by sorry

end cube_edge_sum_l1743_174311


namespace unique_polygon_pair_existence_l1743_174307

theorem unique_polygon_pair_existence : 
  ∃! (n₁ n₂ : ℕ), 
    n₁ > 0 ∧ n₂ > 0 ∧
    ∃ x : ℝ, x > 0 ∧
      (180 - 360 / n₁ : ℝ) = x ∧
      (180 - 360 / n₂ : ℝ) = x / 2 :=
by sorry

end unique_polygon_pair_existence_l1743_174307


namespace angle_equality_from_cofunctions_l1743_174328

-- Define a type for angles
variable {α : Type*} [AddCommGroup α]

-- Define a function for co-functions (abstract representation)
variable (cofunc : α → ℝ)

-- State the theorem
theorem angle_equality_from_cofunctions (θ₁ θ₂ : α) :
  (θ₁ = θ₂) ∨ (cofunc θ₁ = cofunc θ₂) → θ₁ = θ₂ := by
  sorry

end angle_equality_from_cofunctions_l1743_174328


namespace equal_chord_lengths_l1743_174327

/-- The ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

/-- The first line -/
def line1 (k x y : ℝ) : Prop := k * x + y - 2 = 0

/-- The second line -/
def line2 (k x y : ℝ) : Prop := y = k * x + 2

/-- Length of the chord intercepted by a line on the ellipse -/
noncomputable def chord_length (line : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem equal_chord_lengths (k : ℝ) :
  chord_length (line1 k) = chord_length (line2 k) :=
sorry

end equal_chord_lengths_l1743_174327


namespace factor_expression_l1743_174366

theorem factor_expression (x : ℝ) : 4*x*(x+2) + 9*(x+2) = (x+2)*(4*x+9) := by
  sorry

end factor_expression_l1743_174366


namespace smallest_c_value_l1743_174334

theorem smallest_c_value (c d : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - c*x^2 + d*x - 2550 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  r₁ * r₂ * r₃ = 2550 →
  c = r₁ + r₂ + r₃ →
  c ≥ 42 :=
by sorry

end smallest_c_value_l1743_174334


namespace tiffany_bag_difference_l1743_174336

/-- Calculates the difference in bags between Tuesday and Monday after giving away some bags -/
def bagDifference (mondayBags tuesdayFound givenAway : ℕ) : ℕ :=
  (mondayBags + tuesdayFound - givenAway) - mondayBags

theorem tiffany_bag_difference :
  bagDifference 7 12 4 = 8 := by
  sorry

end tiffany_bag_difference_l1743_174336


namespace monotonic_increasing_interval_l1743_174320

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem monotonic_increasing_interval (ω : ℝ) (h_pos : ω > 0) (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) := by sorry

end monotonic_increasing_interval_l1743_174320


namespace quadratic_function_properties_l1743_174317

theorem quadratic_function_properties (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : a^2 + 2*a*c + c^2 < b^2) 
  (h3 : ∀ t : ℝ, a*(t+2)^2 + b*(t+2) + c = a*(-t+2)^2 + b*(-t+2) + c) 
  (h4 : a*(-2)^2 + b*(-2) + c = 2) :
  (∃ axis : ℝ, axis = 2 ∧ 
    ∀ x : ℝ, a*x^2 + b*x + c = a*(2*axis - x)^2 + b*(2*axis - x) + c) ∧ 
  (2/15 < a ∧ a < 2/7) := by sorry

end quadratic_function_properties_l1743_174317


namespace find_y_value_l1743_174342

theorem find_y_value (x y : ℝ) (h1 : x * y = 4) (h2 : x / y = 81) (h3 : x > 0) (h4 : y > 0) :
  y = 2 / 9 := by
sorry

end find_y_value_l1743_174342


namespace triangle_value_l1743_174351

theorem triangle_value (triangle p : ℝ) 
  (eq1 : 2 * triangle + p = 72)
  (eq2 : triangle + p + 2 * triangle = 128) :
  triangle = 56 := by
sorry

end triangle_value_l1743_174351


namespace decimal_expansion_18_37_l1743_174357

/-- The decimal expansion of 18/37 has a repeating pattern of length 3 -/
def decimal_expansion_period (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧
  (18 : ℚ) / 37 = (a * 100 + b * 10 + c : ℚ) / 999

/-- The 123rd digit after the decimal point in the expansion of 18/37 -/
def digit_123 : ℕ := 6

theorem decimal_expansion_18_37 :
  decimal_expansion_period 3 ∧ digit_123 = 6 :=
sorry

end decimal_expansion_18_37_l1743_174357


namespace slide_boys_count_l1743_174373

/-- The number of boys who went down the slide initially -/
def initial_boys : ℕ := 22

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := initial_boys + additional_boys

theorem slide_boys_count : total_boys = 35 := by
  sorry

end slide_boys_count_l1743_174373


namespace quartic_equation_minimum_l1743_174315

theorem quartic_equation_minimum (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  a^2 + b^2 ≥ 4/5 := by
sorry

end quartic_equation_minimum_l1743_174315


namespace P_inf_zero_no_minimum_l1743_174332

/-- The function P from ℝ² to ℝ -/
def P : ℝ × ℝ → ℝ := fun (x₁, x₂) ↦ x₁^2 + (1 - x₁ * x₂)^2

theorem P_inf_zero_no_minimum :
  (∀ ε > 0, ∃ x : ℝ × ℝ, P x < ε) ∧
  ¬∃ x : ℝ × ℝ, ∀ y : ℝ × ℝ, P x ≤ P y :=
by sorry

end P_inf_zero_no_minimum_l1743_174332


namespace base5_product_correct_l1743_174384

/-- Converts a base 5 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The first number in base 5 --/
def num1 : List Nat := [3, 0, 2]

/-- The second number in base 5 --/
def num2 : List Nat := [4, 1]

/-- The expected product in base 5 --/
def expected_product : List Nat := [2, 0, 4, 3]

theorem base5_product_correct :
  toBase5 (toDecimal num1 * toDecimal num2) = expected_product := by
  sorry

end base5_product_correct_l1743_174384


namespace dice_roll_sum_l1743_174312

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 360 →
  a + b + c + d ≠ 17 :=
by sorry

end dice_roll_sum_l1743_174312


namespace octagon_ratio_l1743_174361

/-- Represents an octagon with specific properties -/
structure Octagon where
  total_area : ℝ
  unit_squares : ℕ
  pq_divides_equally : Prop
  below_pq_square : ℝ
  below_pq_triangle_base : ℝ
  xq_plus_qy : ℝ

/-- The theorem to be proved -/
theorem octagon_ratio (o : Octagon) 
  (h1 : o.total_area = 12)
  (h2 : o.unit_squares = 12)
  (h3 : o.pq_divides_equally)
  (h4 : o.below_pq_square = 1)
  (h5 : o.below_pq_triangle_base = 6)
  (h6 : o.xq_plus_qy = 6) :
  ∃ (xq qy : ℝ), xq / qy = 2 ∧ xq + qy = o.xq_plus_qy :=
sorry

end octagon_ratio_l1743_174361


namespace transform_458_to_14_l1743_174343

def double (n : ℕ) : ℕ := 2 * n

def eraseLast (n : ℕ) : ℕ := n / 10

inductive Operation
| Double
| EraseLast

def applyOperation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.Double => double n
  | Operation.EraseLast => eraseLast n

def applyOperations (ops : List Operation) (start : ℕ) : ℕ :=
  ops.foldl (fun n op => applyOperation op n) start

theorem transform_458_to_14 :
  ∃ (ops : List Operation), applyOperations ops 458 = 14 :=
sorry

end transform_458_to_14_l1743_174343


namespace ellipse_intersection_equidistant_point_range_l1743_174386

/-- Ellipse G with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : a > 0
  i : b > 0
  j : a > b
  k : e = Real.sqrt 3 / 3
  l : a = Real.sqrt 3

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  k : ℝ
  m : ℝ := 1

/-- Point on x-axis equidistant from intersection points -/
structure EquidistantPoint where
  x : ℝ

/-- Main theorem -/
theorem ellipse_intersection_equidistant_point_range
  (G : Ellipse)
  (l : IntersectingLine)
  (M : EquidistantPoint) :
  (∃ A B : ℝ × ℝ,
    (A.1^2 / G.a^2 + A.2^2 / G.b^2 = 1) ∧
    (B.1^2 / G.a^2 + B.2^2 / G.b^2 = 1) ∧
    (A.2 = l.k * A.1 + l.m) ∧
    (B.2 = l.k * B.1 + l.m) ∧
    ((A.1 - M.x)^2 + A.2^2 = (B.1 - M.x)^2 + B.2^2) ∧
    (M.x ≠ A.1) ∧ (M.x ≠ B.1)) →
  -Real.sqrt 6 / 12 ≤ M.x ∧ M.x ≤ Real.sqrt 6 / 12 :=
by sorry

end ellipse_intersection_equidistant_point_range_l1743_174386


namespace problem_statement_l1743_174383

theorem problem_statement (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = {a + b, 0, a^2} → a^2016 + b^2016 = 1 := by
  sorry

end problem_statement_l1743_174383


namespace largest_digit_sum_for_special_fraction_l1743_174349

/-- A digit is a natural number between 0 and 9 inclusive -/
def Digit := {n : ℕ // n ≤ 9}

/-- abc represents a three-digit number -/
def ThreeDigitNumber (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

theorem largest_digit_sum_for_special_fraction :
  ∃ (a b c : Digit) (y : ℕ),
    (10 ≤ y ∧ y ≤ 99) ∧
    (ThreeDigitNumber a b c : ℚ) / 1000 = 1 / y ∧
    ∀ (a' b' c' : Digit) (y' : ℕ),
      (10 ≤ y' ∧ y' ≤ 99) →
      (ThreeDigitNumber a' b' c' : ℚ) / 1000 = 1 / y' →
      a.val + b.val + c.val ≥ a'.val + b'.val + c'.val ∧
      a.val + b.val + c.val = 7 :=
sorry

end largest_digit_sum_for_special_fraction_l1743_174349


namespace polynomial_division_remainder_l1743_174395

theorem polynomial_division_remainder (x : ℝ) : 
  x^1004 % ((x^2 + 1) * (x - 1)) = x^2 := by
sorry

end polynomial_division_remainder_l1743_174395


namespace tory_has_six_games_l1743_174381

/-- The number of video games Theresa, Julia, and Tory have. -/
structure VideoGames where
  theresa : ℕ
  julia : ℕ
  tory : ℕ

/-- The conditions given in the problem. -/
def problem_conditions (vg : VideoGames) : Prop :=
  vg.theresa = 3 * vg.julia + 5 ∧
  vg.julia = vg.tory / 3 ∧
  vg.theresa = 11

/-- The theorem stating that Tory has 6 video games. -/
theorem tory_has_six_games (vg : VideoGames) (h : problem_conditions vg) : vg.tory = 6 := by
  sorry

end tory_has_six_games_l1743_174381


namespace overall_percentage_increase_l1743_174376

def initial_price_A : ℝ := 300
def initial_price_B : ℝ := 150
def initial_price_C : ℝ := 50
def initial_price_D : ℝ := 100

def new_price_A : ℝ := 390
def new_price_B : ℝ := 180
def new_price_C : ℝ := 70
def new_price_D : ℝ := 110

def total_initial_price : ℝ := initial_price_A + initial_price_B + initial_price_C + initial_price_D
def total_new_price : ℝ := new_price_A + new_price_B + new_price_C + new_price_D

theorem overall_percentage_increase :
  (total_new_price - total_initial_price) / total_initial_price * 100 = 25 := by
  sorry

end overall_percentage_increase_l1743_174376


namespace inequality_solution_l1743_174372

theorem inequality_solution (α x : ℝ) : α * x^2 - 2 ≥ 2 * x - α * x ↔
  (α = 0 ∧ x ≤ -1) ∨
  (α > 0 ∧ (x ≥ 2 / α ∨ x ≤ -1)) ∨
  (-2 < α ∧ α < 0 ∧ 2 / α ≤ x ∧ x ≤ -1) ∨
  (α = -2 ∧ x = -1) ∨
  (α < -2 ∧ -1 ≤ x ∧ x ≤ 2 / α) :=
by sorry

end inequality_solution_l1743_174372


namespace handshakes_in_specific_tournament_l1743_174360

/-- Represents a tennis tournament with the given conditions -/
structure TennisTournament where
  num_teams : Nat
  players_per_team : Nat
  abstaining_player : Nat
  abstained_team : Nat

/-- Calculates the number of handshakes in the tournament -/
def count_handshakes (t : TennisTournament) : Nat :=
  sorry

/-- Theorem stating the number of handshakes in the specific tournament scenario -/
theorem handshakes_in_specific_tournament :
  ∀ (t : TennisTournament),
    t.num_teams = 4 ∧
    t.players_per_team = 2 ∧
    t.abstaining_player ≥ 1 ∧
    t.abstaining_player ≤ 8 ∧
    t.abstained_team ≥ 1 ∧
    t.abstained_team ≤ 4 ∧
    t.abstained_team ≠ ((t.abstaining_player - 1) / 2 + 1) →
    count_handshakes t = 22 :=
  sorry

end handshakes_in_specific_tournament_l1743_174360


namespace intersection_x_coordinate_l1743_174359

/-- Given two points A and B on the natural logarithm curve, prove that the x-coordinate
    of the point E, where E is the intersection of a horizontal line through C
    (C divides AB in a 1:3 ratio) and the natural logarithm curve, is 16. -/
theorem intersection_x_coordinate (x₁ x₂ x₃ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) 
  (h₃ : x₁ = 2) (h₄ : x₂ = 32) : 
  let y₁ := Real.log x₁
  let y₂ := Real.log x₂
  let yC := (1 / 4 : ℝ) * y₁ + (3 / 4 : ℝ) * y₂
  x₃ = Real.exp yC → x₃ = 16 := by
  sorry

end intersection_x_coordinate_l1743_174359


namespace ariel_age_l1743_174367

/-- Ariel's present age in years -/
def present_age : ℕ := 5

/-- The number of years in the future -/
def years_future : ℕ := 15

/-- Theorem stating that Ariel's present age is 5, given the condition -/
theorem ariel_age : 
  (present_age + years_future = 4 * present_age) → present_age = 5 := by
  sorry

end ariel_age_l1743_174367


namespace sisters_name_length_sisters_name_length_is_five_l1743_174353

theorem sisters_name_length (jonathan_first_name_length : ℕ) 
                             (jonathan_surname_length : ℕ) 
                             (sister_surname_length : ℕ) 
                             (total_letters : ℕ) : ℕ :=
  let jonathan_full_name_length := jonathan_first_name_length + jonathan_surname_length
  let sister_first_name_length := total_letters - jonathan_full_name_length - sister_surname_length
  sister_first_name_length

theorem sisters_name_length_is_five : 
  sisters_name_length 8 10 10 33 = 5 := by
  sorry

end sisters_name_length_sisters_name_length_is_five_l1743_174353


namespace total_capacity_l1743_174370

/-- The capacity of a circus tent with five seating sections -/
def circus_tent_capacity (regular_section_capacity : ℕ) (special_section_capacity : ℕ) : ℕ :=
  4 * regular_section_capacity + special_section_capacity

/-- Theorem: The circus tent can accommodate 1298 people -/
theorem total_capacity : circus_tent_capacity 246 314 = 1298 := by
  sorry

end total_capacity_l1743_174370


namespace factor_implies_b_value_l1743_174304

theorem factor_implies_b_value (a b : ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, a * x^3 + b * x^2 + 1 = (x^2 - x - 1) * (x + c)) →
  b = -2 :=
by sorry

end factor_implies_b_value_l1743_174304


namespace cubic_equation_integer_solutions_l1743_174365

theorem cubic_equation_integer_solutions (a b : ℤ) :
  a^3 + b^3 + 3*a*b = 1 ↔ (b = 1 - a) ∨ (a = -1 ∧ b = -1) := by
  sorry

end cubic_equation_integer_solutions_l1743_174365


namespace infinite_solutions_l1743_174374

theorem infinite_solutions (b : ℝ) : 
  (∀ x, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end infinite_solutions_l1743_174374


namespace solve_salary_problem_l1743_174313

def salary_problem (salary_A salary_B : ℝ) : Prop :=
  salary_A + salary_B = 3000 ∧
  0.05 * salary_A = 0.15 * salary_B

theorem solve_salary_problem :
  ∃ (salary_A : ℝ), salary_problem salary_A (3000 - salary_A) ∧ salary_A = 2250 := by
  sorry

end solve_salary_problem_l1743_174313


namespace james_water_storage_l1743_174382

/-- Represents the water storage problem with different container types --/
structure WaterStorage where
  barrelCount : ℕ
  largeCaskCount : ℕ
  smallCaskCount : ℕ
  largeCaskCapacity : ℕ

/-- Calculates the total water storage capacity --/
def totalCapacity (storage : WaterStorage) : ℕ :=
  let barrelCapacity := 2 * storage.largeCaskCapacity + 3
  let smallCaskCapacity := storage.largeCaskCapacity / 2
  storage.barrelCount * barrelCapacity +
  storage.largeCaskCount * storage.largeCaskCapacity +
  storage.smallCaskCount * smallCaskCapacity

/-- Theorem stating that James' total water storage capacity is 282 gallons --/
theorem james_water_storage :
  let storage : WaterStorage := {
    barrelCount := 4,
    largeCaskCount := 3,
    smallCaskCount := 5,
    largeCaskCapacity := 20
  }
  totalCapacity storage = 282 := by
  sorry

end james_water_storage_l1743_174382


namespace nonagon_diagonals_l1743_174352

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l1743_174352


namespace max_non_managers_l1743_174301

/-- The maximum number of non-managers in a department with 8 managers,
    given that the ratio of managers to non-managers must be greater than 7:24 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) :
  managers = 8 →
  (managers : ℚ) / non_managers > 7 / 24 →
  non_managers ≤ 27 :=
by sorry

end max_non_managers_l1743_174301


namespace johns_leisure_travel_l1743_174331

/-- Calculates the leisure travel distance for John given his car's efficiency,
    work commute details, and total gas consumption. -/
theorem johns_leisure_travel
  (efficiency : ℝ)  -- Car efficiency in miles per gallon
  (work_distance : ℝ)  -- One-way distance to work in miles
  (work_days : ℕ)  -- Number of work days per week
  (total_gas : ℝ)  -- Total gas used per week in gallons
  (h1 : efficiency = 30)  -- Car efficiency is 30 mpg
  (h2 : work_distance = 20)  -- Distance to work is 20 miles each way
  (h3 : work_days = 5)  -- Works 5 days a week
  (h4 : total_gas = 8)  -- Uses 8 gallons of gas per week
  : ℝ :=
  total_gas * efficiency - 2 * work_distance * work_days

#check johns_leisure_travel

end johns_leisure_travel_l1743_174331


namespace unique_base7_digit_divisible_by_13_l1743_174305

/-- Converts a base-7 number of the form 3dd6_7 to base-10 --/
def base7ToBase10 (d : ℕ) : ℕ := 3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : ℕ) : Prop := n % 13 = 0

/-- Represents a base-7 digit --/
def isBase7Digit (d : ℕ) : Prop := d ≤ 6

theorem unique_base7_digit_divisible_by_13 :
  ∃! d : ℕ, isBase7Digit d ∧ isDivisibleBy13 (base7ToBase10 d) ∧ d = 2 := by sorry

end unique_base7_digit_divisible_by_13_l1743_174305


namespace proportion_solution_l1743_174369

theorem proportion_solution (n : ℝ) : n / 1.2 = 5 / 8 → n = 0.75 := by
  sorry

end proportion_solution_l1743_174369


namespace power_of_256_three_fourths_l1743_174333

theorem power_of_256_three_fourths : (256 : ℝ) ^ (3/4) = 64 := by sorry

end power_of_256_three_fourths_l1743_174333


namespace milk_per_serving_in_cups_l1743_174338

/-- Proof that the amount of milk required per serving is 0.5 cups -/
theorem milk_per_serving_in_cups : 
  let ml_per_cup : ℝ := 250
  let total_people : ℕ := 8
  let servings_per_person : ℕ := 2
  let milk_cartons : ℕ := 2
  let ml_per_carton : ℝ := 1000
  
  let total_milk : ℝ := milk_cartons * ml_per_carton
  let total_servings : ℕ := total_people * servings_per_person
  let ml_per_serving : ℝ := total_milk / total_servings
  let cups_per_serving : ℝ := ml_per_serving / ml_per_cup

  cups_per_serving = 0.5 := by
  sorry

end milk_per_serving_in_cups_l1743_174338


namespace mikeys_leaves_l1743_174341

/-- The number of leaves that blew away -/
def leaves_blown_away (initial final : ℕ) : ℕ := initial - final

/-- Proof that 244 leaves blew away -/
theorem mikeys_leaves : leaves_blown_away 356 112 = 244 := by
  sorry

end mikeys_leaves_l1743_174341


namespace smallest_angle_in_3_4_5_ratio_triangle_l1743_174308

theorem smallest_angle_in_3_4_5_ratio_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k →  -- angles are in ratio 3:4:5
  min a (min b c) = 45 :=
by sorry

end smallest_angle_in_3_4_5_ratio_triangle_l1743_174308


namespace complex_equation_solution_l1743_174399

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end complex_equation_solution_l1743_174399


namespace probability_twelve_rolls_last_l1743_174390

/-- The probability of getting the same number on the 12th roll as on the 11th roll,
    given that all previous pairs of consecutive rolls were different. -/
theorem probability_twelve_rolls_last (d : ℕ) (h : d = 6) : 
  (((d - 1) / d) ^ 10 * (1 / d) : ℚ) = 9765625 / 362797056 := by
  sorry

end probability_twelve_rolls_last_l1743_174390


namespace smallest_c_value_l1743_174368

/-- Given a cosine function y = a cos(bx + c) with positive constants a, b, c,
    and maximum at x = 1, the smallest possible value of c is 0. -/
theorem smallest_c_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : ∀ x : ℝ, a * Real.cos (b * x + c) ≤ a * Real.cos (b * 1 + c)) :
    ∃ c' : ℝ, c' ≥ 0 ∧ c' ≤ c ∧ ∀ c'' : ℝ, c'' ≥ 0 → c'' ≤ c → c' ≤ c'' := by
  sorry

end smallest_c_value_l1743_174368


namespace sqrt_equation_solution_l1743_174355

theorem sqrt_equation_solution :
  ∀ y : ℚ, (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2)) = 3) → y = 54 / 23 := by
  sorry

end sqrt_equation_solution_l1743_174355


namespace cafeteria_shirts_l1743_174335

theorem cafeteria_shirts (total : ℕ) (checkered : ℕ) (horizontal : ℕ) (vertical : ℕ) : 
  total = 40 →
  checkered = 7 →
  horizontal = 4 * checkered →
  vertical = total - (checkered + horizontal) →
  vertical = 5 :=
by
  sorry

end cafeteria_shirts_l1743_174335


namespace triangle_angle_measure_l1743_174393

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  a = 3 ∧ b = Real.sqrt 3 ∧ A = π / 3 →
  B = π / 6 := by
sorry

end triangle_angle_measure_l1743_174393


namespace leo_current_weight_l1743_174394

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 98

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 170 - leo_weight

/-- Theorem stating that Leo's current weight is 98 pounds -/
theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 170) →
  leo_weight = 98 := by
sorry

end leo_current_weight_l1743_174394


namespace function_sum_at_one_l1743_174380

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem function_sum_at_one 
  (h1 : is_even f) 
  (h2 : is_odd g) 
  (h3 : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := by
sorry

end function_sum_at_one_l1743_174380


namespace complex_square_i_positive_l1743_174318

theorem complex_square_i_positive (a : ℝ) 
  (h : (Complex.I * (a + Complex.I)^2).re > 0 ∧ (Complex.I * (a + Complex.I)^2).im = 0) : 
  a = -1 := by sorry

end complex_square_i_positive_l1743_174318


namespace classroom_capacity_l1743_174363

/-- Calculates the total number of desks in a classroom with an arithmetic progression of desks per row -/
def totalDesks (rows : ℕ) (firstRowDesks : ℕ) (increment : ℕ) : ℕ :=
  rows * (2 * firstRowDesks + (rows - 1) * increment) / 2

/-- Theorem stating that a classroom with 8 rows, starting with 10 desks and increasing by 2 each row, can seat 136 students -/
theorem classroom_capacity :
  totalDesks 8 10 2 = 136 := by
  sorry

#eval totalDesks 8 10 2

end classroom_capacity_l1743_174363


namespace orange_sale_savings_l1743_174350

/-- Calculates the total savings for a mother's birthday gift based on orange sales. -/
theorem orange_sale_savings 
  (liam_oranges : ℕ) 
  (liam_price : ℚ) 
  (claire_oranges : ℕ) 
  (claire_price : ℚ) 
  (h1 : liam_oranges = 40)
  (h2 : liam_price = 5/2)
  (h3 : claire_oranges = 30)
  (h4 : claire_price = 6/5)
  : ℚ :=
by
  sorry

#check orange_sale_savings

end orange_sale_savings_l1743_174350


namespace boat_speed_difference_l1743_174387

/-- Proves that the boat's speed is 1 km/h greater than the stream current speed --/
theorem boat_speed_difference (V : ℝ) : 
  let S := 1 -- distance in km
  let V₁ := 2*V + 1 -- river current speed in km/h
  let T := 1 -- total time in hours
  ∃ (U : ℝ), -- boat's speed
    U > V ∧ -- boat is faster than stream current
    S / (U - V) - S / (U + V) + S / V₁ = T ∧ -- time equation
    U - V = 1 -- difference in speeds
  := by sorry

end boat_speed_difference_l1743_174387


namespace deer_leap_distance_proof_l1743_174371

/-- The distance the tiger needs to catch the deer -/
def catch_distance : ℝ := 800

/-- The number of tiger leaps behind the deer initially -/
def initial_leaps_behind : ℕ := 50

/-- The number of leaps the tiger takes per minute -/
def tiger_leaps_per_minute : ℕ := 5

/-- The number of leaps the deer takes per minute -/
def deer_leaps_per_minute : ℕ := 4

/-- The distance the tiger covers per leap in meters -/
def tiger_leap_distance : ℝ := 8

/-- The distance the deer covers per leap in meters -/
def deer_leap_distance : ℝ := 5

theorem deer_leap_distance_proof :
  deer_leap_distance = 5 :=
sorry

end deer_leap_distance_proof_l1743_174371


namespace arithmetic_sequence_third_term_l1743_174344

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first five terms of the sequence is 20. -/
def SumFirstFiveTerms (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 20

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : IsArithmeticSequence a)
  (h_sum : SumFirstFiveTerms a) :
  a 3 = 4 := by
  sorry

end arithmetic_sequence_third_term_l1743_174344


namespace max_value_theorem_l1743_174362

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  ∀ (a b : ℝ), 4 * a + 3 * b ≤ 10 → 3 * a + 5 * b ≤ 12 → 2 * a + b ≤ 46 / 11 :=
by
  sorry

end max_value_theorem_l1743_174362


namespace empty_set_problem_l1743_174306

-- Define the sets
def set_A : Set ℝ := {x | x^2 - 4 = 0}
def set_B : Set ℝ := {x | x > 9 ∨ x < 3}
def set_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}
def set_D : Set ℝ := {x | x > 9 ∧ x < 3}

-- Theorem statement
theorem empty_set_problem :
  (set_A.Nonempty) ∧
  (set_B.Nonempty) ∧
  (set_C.Nonempty) ∧
  (set_D = ∅) :=
sorry

end empty_set_problem_l1743_174306


namespace smallest_third_term_geometric_progression_l1743_174391

theorem smallest_third_term_geometric_progression 
  (a b c : ℝ) 
  (arithmetic_prog : a = 7 ∧ c - b = b - a) 
  (geometric_prog : ∃ r : ℝ, r > 0 ∧ (b + 3) = a * r ∧ (c + 22) = (b + 3) * r) :
  c + 22 ≥ 23 + 16 * Real.sqrt 7 := by
  sorry

end smallest_third_term_geometric_progression_l1743_174391


namespace set_b_forms_triangle_l1743_174340

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (5, 6, 10) can form a triangle -/
theorem set_b_forms_triangle : can_form_triangle 5 6 10 := by
  sorry


end set_b_forms_triangle_l1743_174340


namespace maintenance_model_correct_l1743_174319

/-- Linear regression model for device maintenance cost --/
structure MaintenanceModel where
  b : ℝ  -- Slope of the regression line
  a : ℝ  -- Y-intercept of the regression line

/-- Conditions for the maintenance cost model --/
class MaintenanceConditions (model : MaintenanceModel) where
  avg_point : 5.4 = 4 * model.b + model.a
  cost_diff : 8 * model.b + model.a - (7 * model.b + model.a) = 1.1

/-- Theorem stating the correctness of the derived model and its prediction --/
theorem maintenance_model_correct (model : MaintenanceModel) 
  [cond : MaintenanceConditions model] : 
  model.b = 0.55 ∧ model.a = 3.2 ∧ 
  (0.55 * 10 + 3.2 : ℝ) = 8.7 := by
  sorry

#check maintenance_model_correct

end maintenance_model_correct_l1743_174319


namespace probability_second_black_given_first_black_l1743_174396

/-- A bag of balls with white and black colors -/
structure BallBag where
  white : ℕ
  black : ℕ

/-- The probability of drawing a specific color ball given the current state of the bag -/
def drawProbability (bag : BallBag) (isBlack : Bool) : ℚ :=
  if isBlack then
    bag.black / (bag.white + bag.black)
  else
    bag.white / (bag.white + bag.black)

/-- The probability of drawing a black ball in the second draw given a black ball was drawn first -/
def secondBlackGivenFirstBlack (initialBag : BallBag) : ℚ :=
  let bagAfterFirstDraw := BallBag.mk initialBag.white (initialBag.black - 1)
  drawProbability bagAfterFirstDraw true

theorem probability_second_black_given_first_black :
  let initialBag := BallBag.mk 3 2
  secondBlackGivenFirstBlack initialBag = 1/4 := by
  sorry

#eval secondBlackGivenFirstBlack (BallBag.mk 3 2)

end probability_second_black_given_first_black_l1743_174396


namespace square_root_difference_product_l1743_174324

theorem square_root_difference_product : (Real.sqrt 100 + Real.sqrt 9) * (Real.sqrt 100 - Real.sqrt 9) = 91 := by
  sorry

end square_root_difference_product_l1743_174324


namespace initial_cooking_time_is_45_l1743_174302

/-- The recommended cooking time in minutes -/
def recommended_time : ℕ := 5

/-- The remaining cooking time in seconds -/
def remaining_time : ℕ := 255

/-- Conversion factor from minutes to seconds -/
def minutes_to_seconds : ℕ := 60

/-- The initial cooking time in seconds -/
def initial_cooking_time : ℕ := recommended_time * minutes_to_seconds - remaining_time

theorem initial_cooking_time_is_45 : initial_cooking_time = 45 := by
  sorry

end initial_cooking_time_is_45_l1743_174302


namespace prob_only_one_value_l1743_174316

/-- The probability that student A solves the problem -/
def prob_A : ℚ := 1/2

/-- The probability that student B solves the problem -/
def prob_B : ℚ := 1/3

/-- The probability that student C solves the problem -/
def prob_C : ℚ := 1/4

/-- The probability that only one student solves the problem -/
def prob_only_one : ℚ :=
  prob_A * (1 - prob_B) * (1 - prob_C) +
  prob_B * (1 - prob_A) * (1 - prob_C) +
  prob_C * (1 - prob_A) * (1 - prob_B)

theorem prob_only_one_value : prob_only_one = 11/24 := by
  sorry

end prob_only_one_value_l1743_174316


namespace e_pow_f_neg_two_eq_half_l1743_174345

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem e_pow_f_neg_two_eq_half
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_log : ∀ x > 0, f x = Real.log x) :
  Real.exp (f (-2)) = 1/2 := by
  sorry

end e_pow_f_neg_two_eq_half_l1743_174345


namespace smallest_fraction_greater_than_three_fifths_l1743_174347

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction where
  numerator : ℕ
  denominator : ℕ
  num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99
  den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99

/-- The property of being greater than 3/5 -/
def greater_than_three_fifths (f : TwoDigitFraction) : Prop :=
  (f.numerator : ℚ) / f.denominator > 3 / 5

/-- The theorem stating that 59/98 is the smallest fraction greater than 3/5 with two-digit numerator and denominator -/
theorem smallest_fraction_greater_than_three_fifths :
  ∀ f : TwoDigitFraction, greater_than_three_fifths f →
    (59 : ℚ) / 98 ≤ (f.numerator : ℚ) / f.denominator :=
by sorry

end smallest_fraction_greater_than_three_fifths_l1743_174347


namespace contrapositive_equivalence_l1743_174388

theorem contrapositive_equivalence (a b : ℝ) :
  ((a^2 + b^2 = 0) → (a = 0 ∧ b = 0)) ↔ ((a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0)) := by
  sorry

end contrapositive_equivalence_l1743_174388


namespace sqrt_equality_l1743_174329

theorem sqrt_equality (m n : ℝ) (h1 : m > 0) (h2 : 0 ≤ n) (h3 : n ≤ 3*m) :
  Real.sqrt (6*m + 2*Real.sqrt (9*m^2 - n^2)) - Real.sqrt (6*m - 2*Real.sqrt (9*m^2 - n^2)) = 2 * Real.sqrt (3*m - n) := by
  sorry

end sqrt_equality_l1743_174329


namespace inequality_solution_l1743_174354

/-- Given that the solution of the inequality 2x^2 - 6x + 4 < 0 is 1 < x < b, prove that b = 2 -/
theorem inequality_solution (b : ℝ) 
  (h : ∀ x : ℝ, 1 < x ∧ x < b ↔ 2 * x^2 - 6 * x + 4 < 0) : 
  b = 2 := by
sorry

end inequality_solution_l1743_174354


namespace map_distance_conversion_l1743_174322

/-- Proves that given a map scale where 312 inches represents 136 km,
    a point 25 inches away on the map corresponds to approximately 10.9 km
    in actual distance. -/
theorem map_distance_conversion
  (map_distance : ℝ) (actual_distance : ℝ) (point_on_map : ℝ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : point_on_map = 25) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧
  abs ((actual_distance / map_distance) * point_on_map - 10.9) < ε :=
sorry

end map_distance_conversion_l1743_174322


namespace namjoon_rank_l1743_174303

theorem namjoon_rank (total_participants : ℕ) (worse_performers : ℕ) (h1 : total_participants = 13) (h2 : worse_performers = 4) :
  total_participants - worse_performers - 1 = 9 :=
by sorry

end namjoon_rank_l1743_174303


namespace coprime_linear_combination_l1743_174397

theorem coprime_linear_combination (m n : ℕ+) (h : Nat.Coprime m n) :
  ∃ N : ℕ, ∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n ∧
  (∀ N' : ℕ, (∀ k : ℕ, k ≥ N' → ∃ a b : ℕ, k = a * m + b * n) → N' ≥ N) ∧
  N = m * n - m - n + 1 :=
sorry

end coprime_linear_combination_l1743_174397


namespace puzzle_missing_pieces_l1743_174310

/-- Calculates the number of missing puzzle pieces. -/
def missing_pieces (total : ℕ) (border : ℕ) (trevor : ℕ) (joe_multiplier : ℕ) : ℕ :=
  total - (border + trevor + joe_multiplier * trevor)

/-- Proves that the number of missing puzzle pieces is 5. -/
theorem puzzle_missing_pieces :
  missing_pieces 500 75 105 3 = 5 := by
  sorry

end puzzle_missing_pieces_l1743_174310


namespace final_result_l1743_174392

def program_result : ℕ → ℕ → ℕ
| 0, s => s
| (n+1), s => program_result n (s * (11 - n))

theorem final_result : program_result 3 1 = 990 := by
  sorry

#eval program_result 3 1

end final_result_l1743_174392


namespace ratio_of_divisors_sums_l1743_174378

def M : ℕ := 75 * 75 * 140 * 343

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 6 := by sorry

end ratio_of_divisors_sums_l1743_174378


namespace fraction_representation_of_naturals_l1743_174321

theorem fraction_representation_of_naturals (n : ℕ) :
  ∃ x y : ℕ, n = x^3 / y^4 :=
sorry

end fraction_representation_of_naturals_l1743_174321


namespace bird_cage_problem_l1743_174375

theorem bird_cage_problem (N : ℚ) : 
  (5/8 * (4/5 * (1/2 * N + 12) + 20) = 60) → N = 166 := by
  sorry

end bird_cage_problem_l1743_174375


namespace sin_x_bounds_l1743_174389

theorem sin_x_bounds (x : ℝ) (h : 0 < x) (h' : x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by
  sorry

end sin_x_bounds_l1743_174389


namespace intersection_not_roots_l1743_174325

theorem intersection_not_roots : ∀ x : ℝ,
  (x^2 - 1 = x + 7) → (x^2 + x - 6 ≠ 0) := by
  sorry

end intersection_not_roots_l1743_174325


namespace smallest_integer_solution_l1743_174337

theorem smallest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, |y^2 - 5*y + 6| = 14 → x ≤ y) ↔ x = -1 :=
by sorry

end smallest_integer_solution_l1743_174337


namespace complex_number_problem_l1743_174300

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * Complex.I = 1 + Complex.I) →
  (z₂.im = 2) →
  ((z₁ * z₂).im = 0) →
  (z₁ = 3 - Complex.I ∧ z₂ = 6 + 2 * Complex.I) := by
sorry

end complex_number_problem_l1743_174300


namespace cruise_liner_passengers_l1743_174358

theorem cruise_liner_passengers : ∃ n : ℕ, 
  (250 ≤ n ∧ n ≤ 400) ∧ 
  (∃ r : ℕ, n = 15 * r + 7) ∧
  (∃ s : ℕ, n = 25 * s - 8) ∧
  (n = 292 ∨ n = 367) := by
sorry

end cruise_liner_passengers_l1743_174358


namespace f_has_one_or_two_zeros_l1743_174348

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - m^2

-- State the theorem
theorem f_has_one_or_two_zeros (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), f m x₁ = 0 ∧ f m x₂ = 0 ∧ (x₁ = x₂ ∨ x₁ ≠ x₂) :=
sorry

end f_has_one_or_two_zeros_l1743_174348


namespace meadow_business_revenue_l1743_174309

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  boxes_per_week : ℕ
  packs_per_box : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ

/-- Calculates the total money made from selling all diapers --/
def total_money_made (business : DiaperBusiness) : ℕ :=
  business.boxes_per_week * business.packs_per_box * business.diapers_per_pack * business.price_per_diaper

/-- Theorem stating that Meadow's business makes $960000 from selling all diapers --/
theorem meadow_business_revenue :
  let meadow_business : DiaperBusiness := {
    boxes_per_week := 30,
    packs_per_box := 40,
    diapers_per_pack := 160,
    price_per_diaper := 5
  }
  total_money_made meadow_business = 960000 := by
  sorry

end meadow_business_revenue_l1743_174309


namespace sine_shifted_is_even_l1743_174379

/-- A function that reaches its maximum at x = 1 -/
def reaches_max_at_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f x ≤ f 1

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Main theorem -/
theorem sine_shifted_is_even
    (A ω φ : ℝ)
    (hA : A > 0)
    (hω : ω > 0)
    (h_max : reaches_max_at_one (fun x ↦ A * Real.sin (ω * x + φ))) :
    is_even (fun x ↦ A * Real.sin (ω * (x + 1) + φ)) := by
  sorry

end sine_shifted_is_even_l1743_174379
