import Mathlib

namespace cut_tetrahedron_unfolds_to_given_config_l2346_234637

/-- Represents a polyhedron with vertices and edges -/
structure Polyhedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)

/-- Represents the unfolded configuration of a polyhedron -/
structure UnfoldedConfig where
  vertices : Finset (ℝ × ℝ)
  edges : Finset ((ℝ × ℝ) × (ℝ × ℝ))

/-- The given unfolded configuration from the problem -/
def given_config : UnfoldedConfig := sorry

/-- A tetrahedron with a smaller tetrahedron removed -/
def cut_tetrahedron : Polyhedron where
  vertices := {1, 2, 3, 4, 5, 6, 7, 8}
  edges := {(1,2), (1,3), (1,4), (2,3), (2,4), (3,4),
            (1,5), (2,6), (3,7), (4,8),
            (5,6), (6,7), (7,8)}

/-- Function to unfold a polyhedron onto a plane -/
def unfold (p : Polyhedron) : UnfoldedConfig := sorry

theorem cut_tetrahedron_unfolds_to_given_config :
  unfold cut_tetrahedron = given_config := by sorry

end cut_tetrahedron_unfolds_to_given_config_l2346_234637


namespace bug_final_position_l2346_234625

def CirclePoints : Nat := 7

def jump (start : Nat) : Nat :=
  if start % 2 == 0 then
    (start + 2 - 1) % CirclePoints + 1
  else
    (start + 3 - 1) % CirclePoints + 1

def bug_position (start : Nat) (jumps : Nat) : Nat :=
  match jumps with
  | 0 => start
  | n + 1 => jump (bug_position start n)

theorem bug_final_position :
  bug_position 7 2023 = 1 := by sorry

end bug_final_position_l2346_234625


namespace smallest_divisible_by_1_to_12_l2346_234603

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end smallest_divisible_by_1_to_12_l2346_234603


namespace continuous_stripe_probability_l2346_234692

/-- Represents the possible stripe configurations on a cube face -/
inductive StripeConfig
| DiagonalA
| DiagonalB
| EdgeToEdgeA
| EdgeToEdgeB

/-- Represents a cube with stripes on each face -/
def StripedCube := Fin 6 → StripeConfig

/-- Checks if a given StripedCube has a continuous stripe encircling it -/
def hasContinuousStripe (cube : StripedCube) : Prop := sorry

/-- The total number of possible stripe configurations for a cube -/
def totalConfigurations : ℕ := 4^6

/-- The number of configurations that result in a continuous stripe -/
def continuousStripeConfigurations : ℕ := 48

theorem continuous_stripe_probability :
  (continuousStripeConfigurations : ℚ) / totalConfigurations = 3 / 256 := by sorry

end continuous_stripe_probability_l2346_234692


namespace quadratic_inequality_equivalence_l2346_234695

theorem quadratic_inequality_equivalence (x : ℝ) :
  3 * x^2 + 5 * x < 8 ↔ -4 < x ∧ x < 2/3 := by sorry

end quadratic_inequality_equivalence_l2346_234695


namespace arithmetic_mean_difference_l2346_234699

theorem arithmetic_mean_difference (a b c : ℝ) :
  (a + b) / 2 = (a + b + c) / 3 + 5 →
  (a + c) / 2 = (a + b + c) / 3 - 8 →
  (b + c) / 2 = (a + b + c) / 3 + 3 := by
  sorry

end arithmetic_mean_difference_l2346_234699


namespace cubic_root_sum_l2346_234675

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 - 2*a^2 - a + 2 = 0) →
  (b^3 - 2*b^2 - b + 2 = 0) →
  (c^3 - 2*c^2 - c + 2 = 0) →
  a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 = 0 := by
sorry

end cubic_root_sum_l2346_234675


namespace wire_oscillation_period_l2346_234670

/-- The period of oscillation for a mass on a wire with small displacements -/
theorem wire_oscillation_period
  (l d g : ℝ) -- wire length, distance between fixed points, gravitational acceleration
  (G : ℝ) -- mass
  (h_l_pos : l > 0)
  (h_d_pos : d > 0)
  (h_g_pos : g > 0)
  (h_G_pos : G > 0)
  (h_l_gt_d : l > d)
  (h_small_displacements : True) -- Assumption for small displacements
  : ∃ (T : ℝ), T = Real.pi * l * Real.sqrt (Real.sqrt 2 / (g * Real.sqrt (l^2 - d^2))) :=
sorry

end wire_oscillation_period_l2346_234670


namespace smallest_b_value_l2346_234679

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 7) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val^2 * b.val) = 12) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, 
    a'.val - k.val = 7 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val^2 * k.val) = 12) :=
sorry

end smallest_b_value_l2346_234679


namespace f_of_one_eq_zero_l2346_234617

theorem f_of_one_eq_zero (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2*x) : f 1 = 0 := by
  sorry

end f_of_one_eq_zero_l2346_234617


namespace smallest_triple_consecutive_sum_l2346_234658

def sum_of_consecutive (n : ℕ) (k : ℕ) : ℕ := 
  k * n + k * (k - 1) / 2

def is_sum_of_consecutive (x : ℕ) (k : ℕ) : Prop :=
  ∃ n : ℕ, sum_of_consecutive n k = x

theorem smallest_triple_consecutive_sum : 
  (∀ m : ℕ, m < 105 → ¬(is_sum_of_consecutive m 5 ∧ is_sum_of_consecutive m 6 ∧ is_sum_of_consecutive m 7)) ∧ 
  (is_sum_of_consecutive 105 5 ∧ is_sum_of_consecutive 105 6 ∧ is_sum_of_consecutive 105 7) :=
sorry

end smallest_triple_consecutive_sum_l2346_234658


namespace arithmetic_sequence_150th_term_l2346_234663

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 150th term of the specific arithmetic sequence is 1046 -/
theorem arithmetic_sequence_150th_term :
  arithmetic_sequence 3 7 150 = 1046 := by
  sorry

end arithmetic_sequence_150th_term_l2346_234663


namespace geometric_series_ratio_l2346_234600

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
  sorry

end geometric_series_ratio_l2346_234600


namespace rhombus_line_equations_l2346_234631

-- Define the rhombus ABCD
structure Rhombus where
  A : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

-- Define the rhombus with given coordinates
def given_rhombus : Rhombus := {
  A := (-4, 7)
  C := (2, -3)
  P := (3, -1)
}

-- Define a line equation
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem statement
theorem rhombus_line_equations (ABCD : Rhombus) 
  (h1 : ABCD = given_rhombus) :
  ∃ (line_AD line_BD : LineEquation),
    (line_AD.a = 2 ∧ line_AD.b = -1 ∧ line_AD.c = 15) ∧
    (line_BD.a = 3 ∧ line_BD.b = -5 ∧ line_BD.c = 13) := by
  sorry

end rhombus_line_equations_l2346_234631


namespace ordering_of_expressions_l2346_234667

theorem ordering_of_expressions : 
  Real.exp 0.1 > Real.sqrt 1.2 ∧ Real.sqrt 1.2 > 1 + Real.log 1.1 := by
  sorry

end ordering_of_expressions_l2346_234667


namespace tunneled_cube_surface_area_l2346_234642

/-- Represents a cube with a tunnel carved through it -/
structure TunneledCube where
  side_length : ℝ
  tunnel_distance : ℝ

/-- Calculates the total surface area of a tunneled cube -/
def total_surface_area (c : TunneledCube) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific tunneled cube -/
theorem tunneled_cube_surface_area :
  ∃ (c : TunneledCube), c.side_length = 10 ∧ c.tunnel_distance = 3 ∧
  total_surface_area c = 600 + 73.5 * Real.sqrt 3 := by
  sorry

end tunneled_cube_surface_area_l2346_234642


namespace delores_initial_money_l2346_234630

/-- The initial amount of money Delores had --/
def initial_amount : ℕ := sorry

/-- The cost of the computer --/
def computer_cost : ℕ := 400

/-- The cost of the printer --/
def printer_cost : ℕ := 40

/-- The amount of money left after purchases --/
def remaining_money : ℕ := 10

/-- Theorem stating that Delores' initial amount of money was $450 --/
theorem delores_initial_money : 
  initial_amount = computer_cost + printer_cost + remaining_money := by sorry

end delores_initial_money_l2346_234630


namespace cos_675_degrees_l2346_234696

theorem cos_675_degrees : Real.cos (675 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_675_degrees_l2346_234696


namespace brothers_initial_money_l2346_234643

theorem brothers_initial_money (michael_initial : ℕ) (brother_final : ℕ) (candy_cost : ℕ) :
  michael_initial = 42 →
  brother_final = 35 →
  candy_cost = 3 →
  ∃ (brother_initial : ℕ),
    brother_initial + michael_initial / 2 = brother_final + candy_cost ∧
    brother_initial = 17 :=
by sorry

end brothers_initial_money_l2346_234643


namespace set_A_definition_l2346_234601

def U : Set ℝ := {x | x > 1}

theorem set_A_definition (A : Set ℝ) (h1 : A ⊆ U) (h2 : (U \ A) = {x | x > 9}) : 
  A = {x | 1 < x ∧ x ≤ 9} := by
  sorry

end set_A_definition_l2346_234601


namespace quadratic_equation_properties_l2346_234624

/-- The discriminant of the quadratic equation x² - (m + 3)x + m + 1 = 0 -/
def discriminant (m : ℝ) : ℝ := (m + 1)^2 + 4

theorem quadratic_equation_properties :
  (∀ m : ℝ, discriminant m > 0) ∧
  ({m : ℝ | discriminant m = 5} = {0, -2}) := by
  sorry

#check quadratic_equation_properties

end quadratic_equation_properties_l2346_234624


namespace swimming_pool_containers_l2346_234672

/-- The minimum number of containers needed to fill a pool -/
def min_containers (pool_capacity : ℕ) (container_capacity : ℕ) : ℕ :=
  (pool_capacity + container_capacity - 1) / container_capacity

/-- Theorem: 30 containers of 75 liters each are needed to fill a 2250-liter pool -/
theorem swimming_pool_containers : 
  min_containers 2250 75 = 30 := by
  sorry

end swimming_pool_containers_l2346_234672


namespace perpendicular_condition_l2346_234611

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = ax + 1 -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x - 1 -/
def slope2 (a : ℝ) : ℝ := a - 2

/-- Theorem: a = 1 is the necessary and sufficient condition for the lines to be perpendicular -/
theorem perpendicular_condition : 
  ∀ a : ℝ, perpendicular (slope1 a) (slope2 a) ↔ a = 1 := by sorry

end perpendicular_condition_l2346_234611


namespace emily_big_garden_seeds_l2346_234632

def emily_garden_problem (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem emily_big_garden_seeds :
  emily_garden_problem 42 3 2 = 36 := by
  sorry

end emily_big_garden_seeds_l2346_234632


namespace geometry_propositions_l2346_234656

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) :
  (∀ m α β, parallel_line_plane m α → perpendicular_line_plane m β → perpendicular_plane α β) ∧
  (∀ m n α, parallel_line m n → perpendicular_line_plane m α → perpendicular_line_plane n α) :=
sorry

end geometry_propositions_l2346_234656


namespace largest_common_term_l2346_234676

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def is_common_term (x : ℤ) (a₁ d₁ a₂ d₂ : ℤ) : Prop :=
  ∃ n m : ℕ, arithmetic_sequence a₁ d₁ n = x ∧ arithmetic_sequence a₂ d₂ m = x

theorem largest_common_term :
  ∃ x : ℤ, x ≤ 150 ∧ is_common_term x 1 8 5 9 ∧
    ∀ y : ℤ, y ≤ 150 → is_common_term y 1 8 5 9 → y ≤ x :=
sorry

end largest_common_term_l2346_234676


namespace triangle_perimeters_l2346_234623

/-- The possible side lengths of the triangle -/
def triangle_sides : Set ℝ := {3, 6}

/-- Check if three numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible perimeters of the triangle -/
def possible_perimeters : Set ℝ := {9, 15, 18}

/-- Theorem stating that the possible perimeters are 9, 15, or 18 -/
theorem triangle_perimeters :
  ∀ a b c : ℝ,
  a ∈ triangle_sides → b ∈ triangle_sides → c ∈ triangle_sides →
  is_triangle a b c →
  a + b + c ∈ possible_perimeters :=
sorry

end triangle_perimeters_l2346_234623


namespace scaled_building_height_l2346_234674

/-- Calculates the height of a scaled model building given the original building's height and the volumes of water held in the top portions of both the original and the model. -/
theorem scaled_building_height
  (original_height : ℝ)
  (original_volume : ℝ)
  (model_volume : ℝ)
  (h_original_height : original_height = 120)
  (h_original_volume : original_volume = 30000)
  (h_model_volume : model_volume = 0.03)
  : ∃ (model_height : ℝ), model_height = 1.2 := by
  sorry

end scaled_building_height_l2346_234674


namespace unique_three_digit_reborn_number_l2346_234659

def is_reborn_number (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    (a ≠ b ∨ b ≠ c ∨ a ≠ c) ∧
    n = (100 * max a (max b c) + 10 * max (min a b) (max (min a c) (min b c)) + min a (min b c)) -
        (100 * min a (min b c) + 10 * min (max a b) (min (max a c) (max b c)) + max a (max b c))

theorem unique_three_digit_reborn_number :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ is_reborn_number n ↔ n = 495 := by
  sorry

end unique_three_digit_reborn_number_l2346_234659


namespace net_folds_to_partial_cube_l2346_234607

/-- Represents a net that can be folded into a cube -/
structure Net where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  holes : Finset (Fin 12)

/-- Represents a partial cube -/
structure PartialCube where
  faces : Finset (Fin 6)
  edges : Finset (Fin 12)
  holes : Finset (Fin 12)

/-- A net can be folded into a partial cube -/
def canFoldInto (n : Net) (pc : PartialCube) : Prop :=
  n.faces = pc.faces ∧ n.edges = pc.edges ∧ n.holes = pc.holes

/-- The given partial cube has holes on the edges of four different faces -/
axiom partial_cube_property (pc : PartialCube) :
  ∃ (f1 f2 f3 f4 : Fin 6) (e1 e2 e3 e4 : Fin 12),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f3 ≠ f4 ∧
    e1 ∈ pc.holes ∧ e2 ∈ pc.holes ∧ e3 ∈ pc.holes ∧ e4 ∈ pc.holes

/-- Theorem: A net can be folded into the given partial cube if and only if
    it has holes on the edges of four different faces -/
theorem net_folds_to_partial_cube (n : Net) (pc : PartialCube) :
  canFoldInto n pc ↔
    ∃ (f1 f2 f3 f4 : Fin 6) (e1 e2 e3 e4 : Fin 12),
      f1 ≠ f2 ∧ f1 ≠ f3 ∧ f1 ≠ f4 ∧ f2 ≠ f3 ∧ f2 ≠ f4 ∧ f3 ≠ f4 ∧
      e1 ∈ n.holes ∧ e2 ∈ n.holes ∧ e3 ∈ n.holes ∧ e4 ∈ n.holes :=
by sorry

end net_folds_to_partial_cube_l2346_234607


namespace nth_term_formula_l2346_234610

/-- Represents the coefficient of the nth term in the sequence -/
def coeff (n : ℕ) : ℕ := n + 1

/-- Represents the exponent of 'a' in the nth term of the sequence -/
def exponent (n : ℕ) : ℕ := n

/-- Represents the nth term in the sequence as a function of 'a' -/
def nthTerm (n : ℕ) (a : ℝ) : ℝ := (coeff n : ℝ) * (a ^ exponent n)

/-- The theorem stating that the nth term of the sequence is (n+1)aⁿ -/
theorem nth_term_formula (n : ℕ) (a : ℝ) : nthTerm n a = (n + 1 : ℝ) * a ^ n := by sorry

end nth_term_formula_l2346_234610


namespace decimal_number_problem_l2346_234647

theorem decimal_number_problem :
  ∃ x : ℝ, 
    0 ≤ x ∧ 
    x < 10 ∧ 
    (∃ n : ℤ, ⌊x⌋ = n) ∧
    ⌊x⌋ + 4 * x = 21.2 ∧
    x = 4.3 := by
  sorry

end decimal_number_problem_l2346_234647


namespace circle_intersection_x_coordinate_l2346_234671

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle defined by two points on its diameter -/
structure Circle where
  p1 : Point
  p2 : Point

/-- Function to check if a given x-coordinate is one of the intersection points -/
def isIntersectionPoint (c : Circle) (x : ℝ) : Prop :=
  let center_x := (c.p1.x + c.p2.x) / 2
  let center_y := (c.p1.y + c.p2.y) / 2
  let radius := Real.sqrt ((c.p1.x - center_x)^2 + (c.p1.y - center_y)^2)
  (x - center_x)^2 + (1 - center_y)^2 = radius^2

/-- Theorem stating that one of the intersection points has x-coordinate 3 or 5 -/
theorem circle_intersection_x_coordinate 
  (c : Circle) 
  (h1 : c.p1 = ⟨1, 5⟩) 
  (h2 : c.p2 = ⟨7, 3⟩) : 
  isIntersectionPoint c 3 ∨ isIntersectionPoint c 5 := by
  sorry


end circle_intersection_x_coordinate_l2346_234671


namespace polar_to_rectangular_conversion_l2346_234657

theorem polar_to_rectangular_conversion :
  let r : ℝ := 5
  let θ : ℝ := 5 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -5 * Real.sqrt 2 / 2) ∧ (y = -5 * Real.sqrt 2 / 2) := by sorry

end polar_to_rectangular_conversion_l2346_234657


namespace marble_replacement_l2346_234652

theorem marble_replacement (total : ℕ) (red blue yellow white black : ℕ) : 
  total = red + blue + yellow + white + black →
  red = (40 * total) / 100 →
  blue = (25 * total) / 100 →
  yellow = (10 * total) / 100 →
  white = (15 * total) / 100 →
  black = 20 →
  (blue + red / 3 : ℕ) = 77 := by
  sorry

end marble_replacement_l2346_234652


namespace divisor_problem_l2346_234688

theorem divisor_problem (D : ℕ) : 
  D > 0 ∧
  242 % D = 11 ∧
  698 % D = 18 ∧
  (242 + 698) % D = 9 →
  D = 20 := by
sorry

end divisor_problem_l2346_234688


namespace free_throw_probabilities_l2346_234697

/-- The probability of player A scoring a free throw -/
def prob_A : ℚ := 1/2

/-- The probability of player B scoring a free throw -/
def prob_B : ℚ := 2/5

/-- The probability of both A and B scoring their free throws -/
def prob_both_score : ℚ := prob_A * prob_B

/-- The probability of at least one of A or B scoring their free throw -/
def prob_at_least_one_scores : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

theorem free_throw_probabilities :
  (prob_both_score = 1/5) ∧ (prob_at_least_one_scores = 7/10) := by
  sorry

end free_throw_probabilities_l2346_234697


namespace equal_population_after_15_years_l2346_234641

/-- The rate of population increase in Village Y that results in equal populations after 15 years -/
def rate_of_increase_village_y (
  initial_population_x : ℕ
  ) (initial_population_y : ℕ
  ) (decrease_rate_x : ℕ
  ) (years : ℕ
  ) : ℕ :=
  (initial_population_x - decrease_rate_x * years - initial_population_y) / years

theorem equal_population_after_15_years 
  (initial_population_x : ℕ)
  (initial_population_y : ℕ)
  (decrease_rate_x : ℕ)
  (years : ℕ) :
  initial_population_x = 72000 →
  initial_population_y = 42000 →
  decrease_rate_x = 1200 →
  years = 15 →
  rate_of_increase_village_y initial_population_x initial_population_y decrease_rate_x years = 800 :=
by
  sorry

#eval rate_of_increase_village_y 72000 42000 1200 15

end equal_population_after_15_years_l2346_234641


namespace adam_room_capacity_l2346_234662

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 10

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 8

/-- The total number of action figures Adam's room could hold. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

/-- Theorem stating that the total number of action figures Adam's room could hold is 80. -/
theorem adam_room_capacity : total_figures = 80 := by sorry

end adam_room_capacity_l2346_234662


namespace no_distinct_naturals_satisfying_equation_l2346_234633

theorem no_distinct_naturals_satisfying_equation :
  ¬ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (2 * a + Nat.lcm b c = 2 * b + Nat.lcm a c) ∧
    (2 * b + Nat.lcm a c = 2 * c + Nat.lcm a b) :=
by sorry

end no_distinct_naturals_satisfying_equation_l2346_234633


namespace complex_trig_identity_l2346_234629

theorem complex_trig_identity (θ : Real) (h : π < θ ∧ θ < (3 * π) / 2) :
  Real.sqrt ((1 / 2) + (1 / 2) * Real.sqrt ((1 / 2) + (1 / 2) * Real.cos (2 * θ))) - 
  Real.sqrt (1 - Real.sin θ) = Real.cos (θ / 2) := by
  sorry

end complex_trig_identity_l2346_234629


namespace jennifer_spending_l2346_234638

theorem jennifer_spending (total : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (book_fraction : ℚ) :
  total = 120 →
  sandwich_fraction = 1 / 5 →
  ticket_fraction = 1 / 6 →
  book_fraction = 1 / 2 →
  total - (sandwich_fraction * total + ticket_fraction * total + book_fraction * total) = 16 := by
  sorry

end jennifer_spending_l2346_234638


namespace work_ratio_man_to_boy_l2346_234622

theorem work_ratio_man_to_boy :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  7 * m + 2 * b = 6 * (m + b) →
  m / b = 4 := by
sorry

end work_ratio_man_to_boy_l2346_234622


namespace initial_num_pipes_is_three_l2346_234685

-- Define the fill time for the initial number of pipes
def initial_fill_time : ℝ := 8

-- Define the fill time for two pipes
def two_pipes_fill_time : ℝ := 12

-- Define the number of pipes we want to prove
def target_num_pipes : ℕ := 3

-- Theorem statement
theorem initial_num_pipes_is_three :
  ∃ (n : ℕ), n > 0 ∧
  (1 : ℝ) / initial_fill_time = (n : ℝ) * ((1 : ℝ) / two_pipes_fill_time / 2) ∧
  n = target_num_pipes :=
sorry

end initial_num_pipes_is_three_l2346_234685


namespace arithmetic_square_root_of_sqrt_16_l2346_234653

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l2346_234653


namespace counterexample_25_l2346_234605

theorem counterexample_25 : 
  ¬(¬(Nat.Prime 25) → Nat.Prime (25 + 3)) := by sorry

end counterexample_25_l2346_234605


namespace no_rational_q_exists_l2346_234668

theorem no_rational_q_exists : ¬ ∃ (q : ℚ) (b c : ℚ),
  -- f(x) = x^2 + bx + c is a quadratic trinomial
  -- The coefficients 1, b, and c form a geometric progression with common ratio q
  ((1 = b ∧ b = c * q) ∨ (1 = c * q ∧ c * q = b) ∨ (b = 1 * q ∧ 1 * q = c)) ∧
  -- The difference between the roots of f(x) is q
  (b^2 - 4*c).sqrt = q := by
sorry

end no_rational_q_exists_l2346_234668


namespace cosine_value_from_tangent_half_l2346_234615

theorem cosine_value_from_tangent_half (α : Real) :
  (1 - Real.cos α) / Real.sin α = 3 → Real.cos α = -4/5 := by
  sorry

end cosine_value_from_tangent_half_l2346_234615


namespace jeds_speed_l2346_234690

def speed_limit : ℕ := 50
def speeding_fine_rate : ℕ := 16
def red_light_fine : ℕ := 75
def cellphone_fine : ℕ := 120
def total_fine : ℕ := 826

def non_speeding_fines : ℕ := 2 * red_light_fine + cellphone_fine

theorem jeds_speed :
  ∃ (speed : ℕ),
    speed = speed_limit + (total_fine - non_speeding_fines) / speeding_fine_rate ∧
    speed = 84 := by
  sorry

end jeds_speed_l2346_234690


namespace tank_fill_time_l2346_234684

theorem tank_fill_time (r1 r2 r3 : ℚ) 
  (h1 : r1 = 1 / 18) 
  (h2 : r2 = 1 / 30) 
  (h3 : r3 = -1 / 45) : 
  (1 / (r1 + r2 + r3)) = 15 := by
  sorry

end tank_fill_time_l2346_234684


namespace hyperbola_center_l2346_234645

/-- The center of a hyperbola is the point (h, k) such that the equation of the hyperbola
    can be written in the form ((y-k)/a)² - ((x-h)/b)² = 1 for some non-zero real numbers a and b. -/
def is_center_of_hyperbola (h k : ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ (x y : ℝ), ((3 * y + 3)^2 / 7^2) - ((4 * x - 5)^2 / 3^2) = 1 ↔
                ((y - k) / a)^2 - ((x - h) / b)^2 = 1

/-- The center of the hyperbola (3y+3)²/7² - (4x-5)²/3² = 1 is (5/4, -1). -/
theorem hyperbola_center :
  is_center_of_hyperbola (5/4) (-1) := by
  sorry

end hyperbola_center_l2346_234645


namespace negative_square_cubed_l2346_234683

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l2346_234683


namespace gruia_puzzle_solution_l2346_234604

/-- The number of Gruis (girls) -/
def num_gruis : ℕ := sorry

/-- The number of gruias (pears) -/
def num_gruias : ℕ := sorry

/-- When each Gruia receives one gruia, there is one gruia left over -/
axiom condition1 : num_gruias = num_gruis + 1

/-- When each Gruia receives two gruias, there is a shortage of two gruias -/
axiom condition2 : num_gruias = 2 * num_gruis - 2

theorem gruia_puzzle_solution : num_gruis = 3 ∧ num_gruias = 4 := by sorry

end gruia_puzzle_solution_l2346_234604


namespace calculation_problem_linear_system_solution_l2346_234648

-- Problem 1
theorem calculation_problem : -Real.sqrt 3 + (-5/2)^0 + |1 - Real.sqrt 3| = 0 := by sorry

-- Problem 2
theorem linear_system_solution :
  ∃ (x y : ℝ), 4*x + 3*y = 10 ∧ 3*x + y = 5 ∧ x = 1 ∧ y = 2 := by sorry

end calculation_problem_linear_system_solution_l2346_234648


namespace stratified_sampling_grade12_l2346_234689

theorem stratified_sampling_grade12 (total_students : ℕ) (grade12_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 3600) 
  (h2 : grade12_students = 1500) 
  (h3 : sample_size = 720) :
  (grade12_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 300 := by
  sorry

end stratified_sampling_grade12_l2346_234689


namespace sqrt_fifth_power_cubed_l2346_234620

theorem sqrt_fifth_power_cubed : (((5 : ℝ) ^ (1/2)) ^ 4) ^ (1/2) ^ 3 = 125 := by sorry

end sqrt_fifth_power_cubed_l2346_234620


namespace trig_identity_l2346_234664

theorem trig_identity (α : ℝ) : -Real.sin α + Real.sqrt 3 * Real.cos α = 2 * Real.sin (α + 2 * Real.pi / 3) := by
  sorry

end trig_identity_l2346_234664


namespace space_diagonal_of_rectangular_solid_l2346_234680

theorem space_diagonal_of_rectangular_solid (l w h : ℝ) (hl : l = 12) (hw : w = 4) (hh : h = 3) :
  Real.sqrt (l^2 + w^2 + h^2) = 13 := by
  sorry

end space_diagonal_of_rectangular_solid_l2346_234680


namespace inequality_proof_l2346_234639

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) ≥ 
  2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c) := by
  sorry

end inequality_proof_l2346_234639


namespace min_value_reciprocal_sum_equality_condition_l2346_234682

theorem min_value_reciprocal_sum (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) :
  1/x + 4/y + 9/z ≥ 36 := by
  sorry

theorem equality_condition (x y z : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) :
  1/x + 4/y + 9/z = 36 ↔ x = 1/6 ∧ y = 1/3 ∧ z = 1/2 := by
  sorry

end min_value_reciprocal_sum_equality_condition_l2346_234682


namespace x_minus_y_equals_four_l2346_234626

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end x_minus_y_equals_four_l2346_234626


namespace katie_spent_sixty_dollars_l2346_234661

/-- The amount Katie spent on flowers -/
def katies_spending (flower_cost : ℕ) (roses : ℕ) (daisies : ℕ) : ℕ :=
  flower_cost * (roses + daisies)

/-- Theorem: Katie spent 60 dollars on flowers -/
theorem katie_spent_sixty_dollars : katies_spending 6 5 5 = 60 := by
  sorry

end katie_spent_sixty_dollars_l2346_234661


namespace min_balls_to_draw_l2346_234677

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- Represents the maximum number of balls that can be drawn for each color -/
structure MaxDrawnBalls where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee the desired outcome -/
def minBallsToGuarantee : Nat := 57

/-- The threshold for a single color to be guaranteed -/
def singleColorThreshold : Nat := 12

/-- Theorem stating the minimum number of balls to be drawn -/
theorem min_balls_to_draw (initial : BallCounts) (max_drawn : MaxDrawnBalls) : 
  initial.red = 30 ∧ 
  initial.green = 25 ∧ 
  initial.yellow = 20 ∧ 
  initial.blue = 15 ∧ 
  initial.white = 10 ∧ 
  initial.black = 5 ∧
  max_drawn.red < singleColorThreshold ∧
  max_drawn.green < singleColorThreshold ∧ 
  max_drawn.yellow < singleColorThreshold ∧
  max_drawn.blue < singleColorThreshold ∧
  max_drawn.white < singleColorThreshold ∧
  max_drawn.black < singleColorThreshold ∧
  max_drawn.green % 2 = 0 ∧
  max_drawn.white % 2 = 0 ∧
  max_drawn.green ≤ initial.green ∧
  max_drawn.white ≤ initial.white →
  minBallsToGuarantee = 
    max_drawn.red + max_drawn.green + max_drawn.yellow + 
    max_drawn.blue + max_drawn.white + max_drawn.black + 1 :=
by sorry

end min_balls_to_draw_l2346_234677


namespace triangle_area_proof_l2346_234616

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.sin (ω * x) ^ 2 + 1

theorem triangle_area_proof (ω : ℝ) (A B C : ℝ) (b : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  b = 2 →
  f ω A = 1 →
  2 * Real.sin A = Real.sqrt 3 * Real.sin C →
  ∃ (a c : ℝ), a * b * Real.sin C / 2 = 2 * Real.sqrt 3 := by
  sorry

#check triangle_area_proof

end triangle_area_proof_l2346_234616


namespace num_tables_made_l2346_234640

-- Define the total number of furniture legs
def total_legs : Nat := 40

-- Define the number of chairs
def num_chairs : Nat := 6

-- Define the number of legs per furniture piece
def legs_per_piece : Nat := 4

-- Theorem to prove
theorem num_tables_made : 
  (total_legs - num_chairs * legs_per_piece) / legs_per_piece = 4 := by
  sorry


end num_tables_made_l2346_234640


namespace complex_equation_solution_l2346_234655

theorem complex_equation_solution (z : ℂ) : 
  z * (1 + Complex.I) = Complex.abs (1 - Complex.I * Real.sqrt 3) →
  z = Real.sqrt 2 * (Complex.cos (Real.pi / 4) - Complex.I * Complex.sin (Real.pi / 4)) :=
by sorry

end complex_equation_solution_l2346_234655


namespace remainder_sum_mod_seven_l2346_234693

theorem remainder_sum_mod_seven (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 0 < b ∧ b < 7 ∧ 0 < c ∧ c < 7 →
  (a * b * c) % 7 = 1 →
  (4 * c) % 7 = 5 →
  (5 * b) % 7 = (4 + b) % 7 →
  (a + b + c) % 7 = 2 := by
sorry

end remainder_sum_mod_seven_l2346_234693


namespace find_divisor_l2346_234669

theorem find_divisor (n : ℕ) (added : ℕ) (divisor : ℕ) : 
  (n + added) % divisor = 0 ∧ 
  (∀ m : ℕ, m < added → (n + m) % divisor ≠ 0) →
  divisor = 2 := by
  sorry

end find_divisor_l2346_234669


namespace arithmetic_sequence_property_l2346_234698

/-- In an arithmetic sequence {aₙ}, if a₄ + a₆ + a₈ + a₁₀ = 28, then a₇ = 7 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  (a 4 + a 6 + a 8 + a 10 = 28) →                   -- given condition
  a 7 = 7 :=                                        -- conclusion to prove
by sorry

end arithmetic_sequence_property_l2346_234698


namespace container_volume_ratio_l2346_234660

theorem container_volume_ratio (V1 V2 V3 : ℚ) : 
  (3/7 : ℚ) * V1 = V2 →  -- First container's juice fills second container
  (3/5 : ℚ) * V3 + (2/3 : ℚ) * ((3/7 : ℚ) * V1) = (4/5 : ℚ) * V3 →  -- Third container's final state
  V1 / V2 = 7/3 := by
sorry

end container_volume_ratio_l2346_234660


namespace arithmetic_sequence_common_difference_l2346_234646

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : 0 < d ∧ d < 1) 
  (h3 : ∀ k : ℤ, a 5 ≠ k * (π / 2)) 
  (h4 : Real.sin (a 3) ^ 2 + 2 * Real.sin (a 5) * Real.cos (a 5) = Real.sin (a 7) ^ 2) : 
  d = π / 8 := by
  sorry

end arithmetic_sequence_common_difference_l2346_234646


namespace arithmetic_mean_two_digit_multiples_of_eight_l2346_234614

theorem arithmetic_mean_two_digit_multiples_of_eight : 
  let first_multiple := 16
  let last_multiple := 96
  let number_of_multiples := (last_multiple - first_multiple) / 8 + 1
  (first_multiple + last_multiple) / 2 = 56 := by
  sorry

end arithmetic_mean_two_digit_multiples_of_eight_l2346_234614


namespace union_and_complement_l2346_234654

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {2, 7}

theorem union_and_complement :
  (A ∪ B = {2, 4, 5, 7}) ∧ (Aᶜ = {1, 3, 6, 7}) := by
  sorry

end union_and_complement_l2346_234654


namespace fitness_center_member_ratio_l2346_234649

theorem fitness_center_member_ratio 
  (f : ℕ) (m : ℕ) -- f: number of female members, m: number of male members
  (h1 : (35 * f + 30 * m) / (f + m) = 32) : -- average age of all members is 32
  f / m = 2 / 3 := by
sorry


end fitness_center_member_ratio_l2346_234649


namespace circle_center_and_radius_l2346_234602

/-- Given a circle with equation x^2 + y^2 + 4x - 2y - 4 = 0, 
    its center coordinates are (-2, 1) and its radius is 3 -/
theorem circle_center_and_radius : 
  ∃ (x y : ℝ), x^2 + y^2 + 4*x - 2*y - 4 = 0 → 
  ∃ (h k r : ℝ), h = -2 ∧ k = 1 ∧ r = 3 ∧
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 := by
sorry

end circle_center_and_radius_l2346_234602


namespace max_n_when_T_less_than_2019_l2346_234619

/-- Define the arithmetic sequence a_n -/
def a (n : ℕ) : ℕ := 2 * n - 1

/-- Define the geometric sequence b_n -/
def b (n : ℕ) : ℕ := 2^(n - 1)

/-- Define the sequence c_n -/
def c (n : ℕ) : ℕ := a (b n)

/-- Define the sum T_n -/
def T (n : ℕ) : ℕ := 2^(n + 1) - n - 2

theorem max_n_when_T_less_than_2019 :
  (∀ n : ℕ, n ≤ 9 → T n < 2019) ∧ T 10 ≥ 2019 := by sorry

end max_n_when_T_less_than_2019_l2346_234619


namespace fraction_sum_simplification_l2346_234627

theorem fraction_sum_simplification (x : ℝ) (h : x + 1 ≠ 0) :
  x / ((x + 1)^2) + 1 / ((x + 1)^2) = 1 / (x + 1) := by
  sorry

end fraction_sum_simplification_l2346_234627


namespace students_not_in_biology_l2346_234612

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) :
  total_students = 840 →
  biology_percentage = 35 / 100 →
  (total_students : ℚ) * (1 - biology_percentage) = 546 := by
  sorry

end students_not_in_biology_l2346_234612


namespace sequence_constant_l2346_234678

theorem sequence_constant (a : ℕ → ℤ) (d : ℤ) :
  (∀ n : ℕ, Nat.Prime (Int.natAbs (a n))) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) + a n + d) →
  (∀ n : ℕ, a n = 0) := by
sorry

end sequence_constant_l2346_234678


namespace fraction_to_repeating_decimal_value_of_expression_l2346_234673

def repeating_decimal (n d : ℕ) (a b c d : ℕ) : Prop :=
  (n : ℚ) / d = (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999

theorem fraction_to_repeating_decimal :
  repeating_decimal 7 26 2 6 9 2 :=
sorry

theorem value_of_expression (a b c d : ℕ) :
  repeating_decimal 7 26 a b c d → 3 * a - b = 0 :=
sorry

end fraction_to_repeating_decimal_value_of_expression_l2346_234673


namespace second_derivative_at_one_l2346_234608

-- Define the function f
def f (x : ℝ) : ℝ := (1 - 2 * x^3)^10

-- State the theorem
theorem second_derivative_at_one (x : ℝ) : 
  (deriv (deriv f)) 1 = 60 := by sorry

end second_derivative_at_one_l2346_234608


namespace reflection_across_x_axis_l2346_234681

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let original : Point2D := { x := 2, y := 3 }
  reflectAcrossXAxis original = { x := 2, y := -3 } := by
  sorry

end reflection_across_x_axis_l2346_234681


namespace lily_siblings_count_l2346_234634

/-- The number of suitcases each sibling brings -/
def suitcases_per_sibling : ℕ := 2

/-- The number of suitcases parents bring -/
def suitcases_parents : ℕ := 6

/-- The total number of suitcases the family brings -/
def total_suitcases : ℕ := 14

/-- The number of Lily's siblings -/
def num_siblings : ℕ := (total_suitcases - suitcases_parents) / suitcases_per_sibling

theorem lily_siblings_count : num_siblings = 4 := by
  sorry

end lily_siblings_count_l2346_234634


namespace cost_per_load_is_25_cents_l2346_234636

/-- Represents the detergent scenario -/
structure DetergentScenario where
  loads_per_bottle : ℕ
  regular_price : ℚ
  sale_price : ℚ

/-- Calculates the cost per load in cents for a given detergent scenario -/
def cost_per_load_cents (scenario : DetergentScenario) : ℚ :=
  (2 * scenario.sale_price * 100) / (2 * scenario.loads_per_bottle)

/-- Theorem stating that the cost per load is 25 cents for the given scenario -/
theorem cost_per_load_is_25_cents (scenario : DetergentScenario)
  (h1 : scenario.loads_per_bottle = 80)
  (h2 : scenario.regular_price = 25)
  (h3 : scenario.sale_price = 20) :
  cost_per_load_cents scenario = 25 := by
  sorry

#eval cost_per_load_cents { loads_per_bottle := 80, regular_price := 25, sale_price := 20 }

end cost_per_load_is_25_cents_l2346_234636


namespace election_win_margin_l2346_234666

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) :
  winner_votes = 1944 →
  (winner_votes : ℚ) / total_votes = 54 / 100 →
  winner_votes - (total_votes - winner_votes) = 288 :=
by
  sorry

end election_win_margin_l2346_234666


namespace intersection_union_problem_l2346_234691

theorem intersection_union_problem (x : ℝ) : 
  let A : Set ℝ := {1, 3, 5}
  let B : Set ℝ := {1, 2, x^2 - 1}
  (A ∩ B = {1, 3}) → (x = -2 ∧ A ∪ B = {1, 2, 3, 5}) := by
sorry

end intersection_union_problem_l2346_234691


namespace ellipse_m_range_l2346_234644

/-- The equation of the curve in Cartesian coordinates -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

/-- The condition for the curve to be an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), curve_equation m x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

/-- The theorem stating the range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m > 5 :=
sorry

end ellipse_m_range_l2346_234644


namespace stones_per_bracelet_l2346_234618

def total_stones : ℝ := 48.0
def num_bracelets : ℕ := 6

theorem stones_per_bracelet :
  total_stones / num_bracelets = 8 := by sorry

end stones_per_bracelet_l2346_234618


namespace f_sum_eq_two_l2346_234687

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.sin x) ^ 2 + b * Real.tan x + 1

theorem f_sum_eq_two (a b : ℝ) (h : f a b 2 = 5) : f a b (Real.pi - 2) + f a b Real.pi = 2 := by
  sorry

end f_sum_eq_two_l2346_234687


namespace quilt_shaded_area_is_40_percent_l2346_234635

/-- Represents a square quilt with shaded areas -/
structure Quilt where
  size : Nat
  fully_shaded : Nat
  half_shaded_single : Nat
  half_shaded_double : Nat

/-- Calculates the percentage of shaded area in the quilt -/
def shaded_percentage (q : Quilt) : Rat :=
  let total_squares := q.size * q.size
  let shaded_area := q.fully_shaded + (q.half_shaded_single / 2) + (q.half_shaded_double / 2)
  (shaded_area / total_squares) * 100

/-- Theorem stating that the specific quilt configuration has 40% shaded area -/
theorem quilt_shaded_area_is_40_percent :
  let q : Quilt := {
    size := 5,
    fully_shaded := 4,
    half_shaded_single := 8,
    half_shaded_double := 4
  }
  shaded_percentage q = 40 := by sorry

end quilt_shaded_area_is_40_percent_l2346_234635


namespace zyx_syndrome_diagnosis_l2346_234613

/-- Represents the characteristics and diagnostic information for ZYX syndrome --/
structure ZYXSyndromeData where
  total_patients : ℕ
  female_ratio : ℚ
  female_syndrome_ratio : ℚ
  male_syndrome_ratio : ℚ
  female_diagnostic_accuracy : ℚ
  male_diagnostic_accuracy : ℚ
  female_false_negative_rate : ℚ
  male_false_negative_rate : ℚ

/-- Calculates the number of patients diagnosed with ZYX syndrome --/
def diagnosed_patients (data : ZYXSyndromeData) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, 14 patients will be diagnosed with ZYX syndrome --/
theorem zyx_syndrome_diagnosis :
  let data : ZYXSyndromeData := {
    total_patients := 52,
    female_ratio := 3/5,
    female_syndrome_ratio := 1/5,
    male_syndrome_ratio := 3/10,
    female_diagnostic_accuracy := 7/10,
    male_diagnostic_accuracy := 4/5,
    female_false_negative_rate := 1/10,
    male_false_negative_rate := 3/20
  }
  diagnosed_patients data = 14 := by
  sorry


end zyx_syndrome_diagnosis_l2346_234613


namespace pigeonhole_principle_buttons_l2346_234650

theorem pigeonhole_principle_buttons : ∀ (r w b : ℕ),
  r ≥ 3 ∧ w ≥ 3 ∧ b ≥ 3 →
  ∀ n : ℕ, n ≥ 7 →
  ∀ f : Fin n → Fin 3,
  ∃ c : Fin 3, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
  f i = c ∧ f j = c ∧ f k = c :=
by
  sorry

#check pigeonhole_principle_buttons

end pigeonhole_principle_buttons_l2346_234650


namespace original_number_proof_l2346_234694

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 → 
  y = 37.66666666666667 → 
  x + y = 32.7 := by
sorry

end original_number_proof_l2346_234694


namespace function_inequality_implies_parameter_range_l2346_234621

theorem function_inequality_implies_parameter_range :
  ∀ (a : ℝ),
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (|x + a| + |x - 2| ≤ |x - 3|)) →
  (a ∈ Set.Icc (-1) 0) :=
by sorry

end function_inequality_implies_parameter_range_l2346_234621


namespace angle_multiplication_l2346_234651

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define multiplication of an angle by a natural number
def multiplyAngle (a : Angle) (n : ℕ) : Angle :=
  let totalMinutes := a.degrees * 60 + a.minutes
  let newTotalMinutes := totalMinutes * n
  ⟨newTotalMinutes / 60, newTotalMinutes % 60⟩

-- Define equality for angles
def angleEq (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes = b.degrees * 60 + b.minutes

-- Theorem statement
theorem angle_multiplication :
  angleEq (multiplyAngle ⟨21, 17⟩ 5) ⟨106, 25⟩ := by
  sorry

end angle_multiplication_l2346_234651


namespace dog_treat_cost_l2346_234628

-- Define the given conditions
def treats_per_day : ℕ := 2
def cost_per_treat : ℚ := 1/10
def days_in_month : ℕ := 30

-- Define the theorem to prove
theorem dog_treat_cost :
  (treats_per_day * days_in_month : ℚ) * cost_per_treat = 6 := by sorry

end dog_treat_cost_l2346_234628


namespace one_real_root_l2346_234686

-- Define the determinant function
def det (x a b d : ℝ) : ℝ := x * (x^2 + a^2 + b^2 + d^2)

-- State the theorem
theorem one_real_root (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃! x : ℝ, det x a b d = 0 :=
sorry

end one_real_root_l2346_234686


namespace bryan_has_more_candies_l2346_234665

/-- Given that Bryan has 50 skittles and Ben has 20 M&M's, 
    prove that Bryan has 30 more candies than Ben. -/
theorem bryan_has_more_candies : 
  ∀ (bryan_skittles ben_mms : ℕ), 
    bryan_skittles = 50 → 
    ben_mms = 20 → 
    bryan_skittles - ben_mms = 30 := by
  sorry

end bryan_has_more_candies_l2346_234665


namespace complex_fraction_simplification_l2346_234606

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * Complex.I
  let z₂ : ℂ := 4 - 6 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 :=
by sorry

end complex_fraction_simplification_l2346_234606


namespace isosceles_obtuse_triangle_smallest_angle_l2346_234609

/-- 
Proves that in an isosceles, obtuse triangle where one angle is 30% larger than a right angle, 
the measure of one of the two smallest angles is 31.5°.
-/
theorem isosceles_obtuse_triangle_smallest_angle : 
  ∀ (a b c : ℝ), 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Isosceles triangle condition
  c > 90 →           -- Obtuse triangle condition
  c = 1.3 * 90 →     -- One angle is 30% larger than a right angle
  a = 31.5 :=        -- The measure of one of the two smallest angles
by
  sorry

end isosceles_obtuse_triangle_smallest_angle_l2346_234609
