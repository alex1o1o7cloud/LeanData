import Mathlib

namespace greater_solution_of_quadratic_l3559_355938

theorem greater_solution_of_quadratic (x : ℝ) :
  x^2 - 5*x - 84 = 0 → x ≤ 12 :=
by
  sorry

end greater_solution_of_quadratic_l3559_355938


namespace randy_money_left_l3559_355959

theorem randy_money_left (initial_amount : ℝ) (lunch_cost : ℝ) (ice_cream_fraction : ℝ) : 
  initial_amount = 30 →
  lunch_cost = 10 →
  ice_cream_fraction = 1/4 →
  initial_amount - lunch_cost - (initial_amount - lunch_cost) * ice_cream_fraction = 15 := by
sorry

end randy_money_left_l3559_355959


namespace first_player_wins_l3559_355926

/-- Represents the state of the game with two piles of tokens -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat)

/-- Defines a valid move in the game -/
inductive ValidMove : GameState → GameState → Prop
  | single_pile (s t : GameState) (i : Fin 2) (n : Nat) :
      n > 0 →
      (i = 0 → t.pile1 = s.pile1 - n ∧ t.pile2 = s.pile2) →
      (i = 1 → t.pile1 = s.pile1 ∧ t.pile2 = s.pile2 - n) →
      ValidMove s t
  | both_piles (s t : GameState) (x y : Nat) :
      x > 0 →
      y > 0 →
      (x + y) % 2015 = 0 →
      t.pile1 = s.pile1 - x →
      t.pile2 = s.pile2 - y →
      ValidMove s t

/-- Defines the winning condition -/
def IsWinningState (s : GameState) : Prop :=
  ∀ t : GameState, ¬ValidMove s t

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → GameState),
    let initial_state := GameState.mk 10000 20000
    ∀ (opponent_move : GameState → GameState),
      ValidMove initial_state (strategy initial_state) →
      (∀ s, ValidMove s (opponent_move s) → ValidMove (opponent_move s) (strategy (opponent_move s))) →
      ∃ n : Nat, IsWinningState (Nat.iterate (λ s => strategy (opponent_move s)) n (strategy initial_state)) :=
sorry

end first_player_wins_l3559_355926


namespace lemonade_second_intermission_l3559_355971

theorem lemonade_second_intermission 
  (total : ℝ) 
  (first : ℝ) 
  (third : ℝ) 
  (h1 : total = 0.9166666666666666) 
  (h2 : first = 0.25) 
  (h3 : third = 0.25) : 
  total - (first + third) = 0.4166666666666666 := by
  sorry

end lemonade_second_intermission_l3559_355971


namespace a_1992_b_1992_values_l3559_355987

def a : ℕ → ℤ
| 0 => 0
| (n + 1) => 2 * a n - a (n - 1) + 2

def b : ℕ → ℤ
| 0 => 8
| (n + 1) => 2 * b n - b (n - 1)

axiom square_sum : ∀ n > 0, ∃ k : ℤ, a n ^ 2 + b n ^ 2 = k ^ 2

theorem a_1992_b_1992_values : 
  (a 1992 = 1992^2 ∧ b 1992 = 7976) ∨ (a 1992 = 1992^2 ∧ b 1992 = -7960) := by
  sorry

end a_1992_b_1992_values_l3559_355987


namespace number_division_problem_l3559_355978

theorem number_division_problem (x : ℝ) : x / 0.04 = 200.9 → x = 8.036 := by
  sorry

end number_division_problem_l3559_355978


namespace log_product_l3559_355984

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_product (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  lg (m * n) = lg m + lg n :=
sorry

end log_product_l3559_355984


namespace smallest_sum_of_c_and_d_l3559_355913

theorem smallest_sum_of_c_and_d (c d : ℝ) (hc : c > 0) (hd : d > 0)
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ 16 * Real.sqrt 3 / 3 := by
sorry

end smallest_sum_of_c_and_d_l3559_355913


namespace arithmetic_mean_of_four_numbers_l3559_355995

theorem arithmetic_mean_of_four_numbers :
  let numbers : List ℝ := [12, 25, 39, 48]
  (numbers.sum / numbers.length : ℝ) = 31 := by
  sorry

end arithmetic_mean_of_four_numbers_l3559_355995


namespace union_A_B_l3559_355974

def A : Set ℝ := {-1, 1}
def B : Set ℝ := {x | x^2 + x - 2 = 0}

theorem union_A_B : A ∪ B = {-2, -1, 1} := by
  sorry

end union_A_B_l3559_355974


namespace array_sum_mod_1004_l3559_355989

/-- Represents the sum of all terms in a 1/q-array as described in the problem -/
def array_sum (q : ℕ) : ℚ :=
  (3 * q^2 : ℚ) / ((3*q - 1) * (q - 1))

/-- The theorem stating that the sum of all terms in a 1/1004-array is congruent to 1 modulo 1004 -/
theorem array_sum_mod_1004 :
  ∃ (n : ℕ), array_sum 1004 = (n * 1004 + 1 : ℚ) := by
  sorry

end array_sum_mod_1004_l3559_355989


namespace bug_crawl_tiles_l3559_355996

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  length : Nat
  width : Nat
  tileSize : Nat
  totalTiles : Nat

/-- Calculates the number of tiles a bug crosses when crawling diagonally across the floor. -/
def tilesTraversed (floor : TiledFloor) : Nat :=
  floor.length + floor.width - 1

/-- Theorem stating the number of tiles crossed by a bug on a specific floor. -/
theorem bug_crawl_tiles (floor : TiledFloor) 
  (h1 : floor.length = 17)
  (h2 : floor.width = 10)
  (h3 : floor.tileSize = 1)
  (h4 : floor.totalTiles = 170) :
  tilesTraversed floor = 26 := by
  sorry

#eval tilesTraversed { length := 17, width := 10, tileSize := 1, totalTiles := 170 }

end bug_crawl_tiles_l3559_355996


namespace rhombus_point_d_y_coord_rhombus_point_d_x_coord_l3559_355908

/-- A rhombus ABCD with specific properties -/
structure Rhombus where
  /-- The y-coordinate of point B -/
  b : ℝ
  /-- The x-coordinate of point D -/
  x : ℝ
  /-- The y-coordinate of point D -/
  y : ℝ
  /-- B is on the negative half of the y-axis -/
  h_b_neg : b < 0
  /-- ABCD is a rhombus -/
  h_is_rhombus : True  -- This is a placeholder, as we can't directly express "is_rhombus" without further definitions
  /-- The intersection of diagonals M is on the x-axis -/
  h_m_on_x_axis : True  -- This is a placeholder, as we can't directly express this geometric property without further definitions

/-- The main theorem about the y-coordinate of point D -/
theorem rhombus_point_d_y_coord (r : Rhombus) : r.y = -r.b - 1 := by sorry

/-- The x-coordinate of point D can be any real number -/
theorem rhombus_point_d_x_coord (r : Rhombus) : r.x ∈ Set.univ := by sorry

end rhombus_point_d_y_coord_rhombus_point_d_x_coord_l3559_355908


namespace deer_bridge_problem_l3559_355979

theorem deer_bridge_problem (y : ℚ) : 
  (3 * (3 * (3 * y - 50) - 50) - 50) * 4 - 50 = 0 ∧ y > 0 → y = 425 / 18 := by
  sorry

end deer_bridge_problem_l3559_355979


namespace sqrt_equation_solution_l3559_355924

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l3559_355924


namespace scientific_notation_of_2102000_l3559_355905

theorem scientific_notation_of_2102000 :
  ∃ (a : ℝ) (n : ℤ), 2102000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.102 ∧ n = 6 :=
by sorry

end scientific_notation_of_2102000_l3559_355905


namespace quadratic_factorization_l3559_355921

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l3559_355921


namespace difference_zero_iff_k_ge_five_l3559_355985

/-- Definition of the sequence u_n -/
def u (n : ℕ) : ℕ := n^4 + 2*n^2

/-- Definition of the first difference operator -/
def Δ₁ (f : ℕ → ℕ) (n : ℕ) : ℕ := f (n + 1) - f n

/-- Definition of the k-th difference operator -/
def Δ (k : ℕ) (f : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => f
  | k + 1 => Δ₁ (Δ k f)

/-- Theorem: The k-th difference of u_n is zero for all n if and only if k ≥ 5 -/
theorem difference_zero_iff_k_ge_five :
  ∀ k : ℕ, (∀ n : ℕ, Δ k u n = 0) ↔ k ≥ 5 :=
sorry

end difference_zero_iff_k_ge_five_l3559_355985


namespace symmetric_complex_product_l3559_355954

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  (z₁.re = z₂.im ∧ z₁.im = z₂.re) →  -- symmetry about y=x
  z₁ * z₂ = Complex.I * 9 →          -- product condition
  Complex.abs z₁ = 3 := by            
  sorry

end symmetric_complex_product_l3559_355954


namespace tangent_sum_simplification_l3559_355909

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (40 * π / 180) + Real.tan (60 * π / 180)) / Real.cos (10 * π / 180) =
  Real.sqrt 3 * ((1/2 * Real.cos (10 * π / 180) + Real.sqrt 3 / 2) / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (40 * π / 180))) := by
  sorry

end tangent_sum_simplification_l3559_355909


namespace distance_walked_when_meeting_l3559_355906

/-- 
Given two people walking towards each other from a distance of 50 miles,
each at a constant speed of 5 miles per hour, prove that one person
will have walked 25 miles when they meet.
-/
theorem distance_walked_when_meeting 
  (initial_distance : ℝ) 
  (speed : ℝ) 
  (h1 : initial_distance = 50)
  (h2 : speed = 5) : 
  (initial_distance / (2 * speed)) * speed = 25 :=
by sorry

end distance_walked_when_meeting_l3559_355906


namespace cube_edge_length_from_circumscribed_sphere_volume_l3559_355968

theorem cube_edge_length_from_circumscribed_sphere_volume 
  (V : ℝ) (a : ℝ) (h : V = (32 / 3) * Real.pi) :
  (V = (4 / 3) * Real.pi * (a * Real.sqrt 3 / 2)^3) → a = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end cube_edge_length_from_circumscribed_sphere_volume_l3559_355968


namespace inscribed_squares_ratio_l3559_355907

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a = 5 → b = 12 → c = 13 → 
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (c - y) = (a - y) * (b - y) →
  x / y = 5 / 13 := by sorry

end inscribed_squares_ratio_l3559_355907


namespace group_frequency_l3559_355955

theorem group_frequency (sample_capacity : ℕ) (group_frequency : ℚ) : 
  sample_capacity = 1000 → 
  group_frequency = 6/10 → 
  (↑sample_capacity * group_frequency : ℚ) = 600 := by
  sorry

end group_frequency_l3559_355955


namespace roots_expression_l3559_355975

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) 
  (hα : α^2 + p*α - 2 = 0)
  (hβ : β^2 + p*β - 2 = 0)
  (hγ : γ^2 + q*γ - 2 = 0)
  (hδ : δ^2 + q*δ - 2 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (q^2 - p^2) := by
  sorry

end roots_expression_l3559_355975


namespace power_of_power_of_three_l3559_355948

theorem power_of_power_of_three :
  (3^3)^(3^3) = 27^27 := by sorry

end power_of_power_of_three_l3559_355948


namespace sphere_in_truncated_cone_l3559_355966

/-- 
Given a sphere perfectly fitted inside a truncated right circular cone, 
if the volume of the truncated cone is three times that of the sphere, 
then the ratio of the radius of the larger base to the radius of the smaller base 
of the truncated cone is (5 + √21) / 2.
-/
theorem sphere_in_truncated_cone (R r s : ℝ) 
  (h_fit : s^2 = R * r)  -- sphere fits perfectly inside the truncated cone
  (h_volume : (π / 3) * (R^2 + R*r + r^2) * (2*s + (2*s*r)/(R-r)) - 
              (π / 3) * r^2 * ((2*s*r)/(R-r)) = 
              4 * π * s^3) :  -- volume relation
  R / r = (5 + Real.sqrt 21) / 2 := by
sorry

end sphere_in_truncated_cone_l3559_355966


namespace total_balls_count_l3559_355997

/-- The number of different colors of balls -/
def num_colors : ℕ := 10

/-- The number of balls for each color -/
def balls_per_color : ℕ := 35

/-- The total number of balls -/
def total_balls : ℕ := num_colors * balls_per_color

theorem total_balls_count : total_balls = 350 := by
  sorry

end total_balls_count_l3559_355997


namespace percentage_problem_l3559_355929

theorem percentage_problem (x : ℝ) (h : 0.8 * x = 240) : 0.2 * x = 60 := by
  sorry

end percentage_problem_l3559_355929


namespace tagged_ratio_is_two_fiftieths_l3559_355988

/-- Represents the fish population and tagging experiment in a pond -/
structure FishExperiment where
  initial_tagged : ℕ
  second_catch : ℕ
  tagged_in_second : ℕ
  total_fish : ℕ

/-- The ratio of tagged fish to total fish in the second catch -/
def tagged_ratio (e : FishExperiment) : ℚ :=
  e.tagged_in_second / e.second_catch

/-- The given experiment data -/
def pond_experiment : FishExperiment :=
  { initial_tagged := 70
  , second_catch := 50
  , tagged_in_second := 2
  , total_fish := 1750 }

/-- Theorem stating that the ratio of tagged fish in the second catch is 2/50 -/
theorem tagged_ratio_is_two_fiftieths :
  tagged_ratio pond_experiment = 2 / 50 := by
  sorry

end tagged_ratio_is_two_fiftieths_l3559_355988


namespace gold_rod_weight_sum_l3559_355970

theorem gold_rod_weight_sum (a : Fin 5 → ℝ) :
  (∀ i j : Fin 5, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
  a 0 = 4 →                                            -- first term is 4
  a 4 = 2 →                                            -- last term is 2
  a 1 + a 3 = 6 :=                                     -- sum of second and fourth terms is 6
by sorry

end gold_rod_weight_sum_l3559_355970


namespace min_positive_period_of_tan_2x_l3559_355925

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem min_positive_period_of_tan_2x :
  ∃ (T : ℝ), T > 0 ∧ T = π / 2 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end min_positive_period_of_tan_2x_l3559_355925


namespace cubic_factorization_l3559_355912

theorem cubic_factorization (x : ℝ) : x^3 + 5*x^2 + 6*x = x*(x+2)*(x+3) := by
  sorry

end cubic_factorization_l3559_355912


namespace sum_of_cubes_equality_l3559_355946

theorem sum_of_cubes_equality (a b c : ℝ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : a^3 + b^3 + c^3 = 3*a*b*c) : 
  a + b + c = 0 := by sorry

end sum_of_cubes_equality_l3559_355946


namespace deepak_present_age_l3559_355911

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age. -/
theorem deepak_present_age 
  (rahul_deepak_ratio : ℚ) 
  (rahul_future_age : ℕ) 
  (years_difference : ℕ) :
  rahul_deepak_ratio = 4 / 3 →
  rahul_future_age = 42 →
  years_difference = 6 →
  ∃ (deepak_age : ℕ), deepak_age = 27 :=
by sorry

end deepak_present_age_l3559_355911


namespace parabola_point_relationship_l3559_355993

/-- Parabola type representing y = ax^2 + bx --/
structure Parabola where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- Point type representing (x, y) coordinates --/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_point_relationship (p : Parabola) (m n t : ℝ) :
  3 * p.a + p.b > 0 →
  p.a + p.b < 0 →
  Point.mk (-3) m ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  Point.mk 2 n ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  Point.mk 4 t ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  n < t ∧ t < m := by
  sorry

end parabola_point_relationship_l3559_355993


namespace sequence_determination_l3559_355963

theorem sequence_determination (p : ℕ) (hp : p.Prime ∧ p > 5) :
  let n := (p - 1) / 2
  ∀ (a : Fin n → ℕ), 
    (∀ i : Fin n, a i ∈ Finset.range n.succ) →
    Function.Injective a →
    (∀ i j : Fin n, i ≠ j → ∃ (r : ℕ), (a i * a j) % p = r) →
    ∃! (b : Fin n → ℕ), ∀ i : Fin n, a i = b i :=
by sorry

end sequence_determination_l3559_355963


namespace min_value_inequality_l3559_355922

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_inequality_l3559_355922


namespace replaced_girl_weight_l3559_355934

theorem replaced_girl_weight 
  (n : ℕ) 
  (new_weight : ℝ) 
  (avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 94)
  (h3 : avg_increase = 3) : 
  ∃ (old_weight : ℝ), 
    old_weight = new_weight - (n * avg_increase) ∧ 
    old_weight = 70 := by
  sorry

end replaced_girl_weight_l3559_355934


namespace pizzeria_sales_l3559_355932

theorem pizzeria_sales (small_price large_price total_revenue small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_revenue = 40)
  (h4 : small_count = 8) :
  ∃ (large_count : ℕ), 
    small_price * small_count + large_price * large_count = total_revenue ∧ 
    large_count = 3 := by
  sorry

end pizzeria_sales_l3559_355932


namespace price_after_discount_l3559_355902

def original_price : ℕ := 76
def discount : ℕ := 25

theorem price_after_discount :
  original_price - discount = 51 := by sorry

end price_after_discount_l3559_355902


namespace quadratic_function_properties_l3559_355957

/-- A quadratic function satisfying certain properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (∃ x₀, f x₀ = -1) ∧
  (f 0 = 0)

theorem quadratic_function_properties (f : ℝ → ℝ) (m : ℝ) 
  (hf : QuadraticFunction f) 
  (h_above : ∀ x ∈ Set.Icc 0 1, f x > 2 * x + 1 + m) :
  (∀ x, f x = x^2 - 2*x) ∧ m < -4 :=
sorry

end quadratic_function_properties_l3559_355957


namespace cos_arithmetic_sequence_product_l3559_355999

theorem cos_arithmetic_sequence_product (a₁ : ℝ) : 
  let a : ℕ+ → ℝ := λ n => a₁ + (2 * π / 3) * (n.val - 1)
  let S : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (a n)}
  (∃ a b : ℝ, S = {a, b} ∧ a ≠ b) → 
  ∃ a b : ℝ, S = {a, b} ∧ a * b = -1/2 :=
by sorry

end cos_arithmetic_sequence_product_l3559_355999


namespace sum_of_roots_is_nine_l3559_355976

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetry property of f
def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

-- Define a property for f having exactly three distinct real roots
def has_three_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, (f x = 0 ∧ f y = 0 ∧ f z = 0) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (∀ w : ℝ, f w = 0 → w = x ∨ w = y ∨ w = z)

-- Theorem statement
theorem sum_of_roots_is_nine (f : ℝ → ℝ) 
  (h1 : is_symmetric_about_3 f) 
  (h2 : has_three_distinct_real_roots f) : 
  ∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0 ∧ x + y + z = 9 :=
sorry

end sum_of_roots_is_nine_l3559_355976


namespace segment_point_relation_l3559_355950

/-- Given a segment AB of length 2 and a point P on AB such that AP² = AB · PB, prove that AP = √5 - 1 -/
theorem segment_point_relation (A B P : ℝ) : 
  (0 ≤ P - A) ∧ (P - A ≤ B - A) ∧  -- P is on segment AB
  (B - A = 2) ∧                    -- AB = 2
  ((P - A)^2 = (B - A) * (B - P))  -- AP² = AB · PB
  → P - A = Real.sqrt 5 - 1 := by sorry

end segment_point_relation_l3559_355950


namespace weight_of_B_l3559_355939

/-- Given the weights of four people A, B, C, and D, prove that B weighs 50 kg. -/
theorem weight_of_B (W_A W_B W_C W_D : ℝ) : W_B = 50 :=
  by
  have h1 : W_A + W_B + W_C + W_D = 240 := by sorry
  have h2 : W_A + W_B = 110 := by sorry
  have h3 : W_B + W_C = 100 := by sorry
  have h4 : W_C + W_D = 130 := by sorry
  sorry

#check weight_of_B

end weight_of_B_l3559_355939


namespace circles_are_separate_l3559_355994

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Equation of the first circle: x² + y² = r₁² -/
  c1 : ℝ → ℝ → Prop
  /-- Equation of the second circle: (x-a)² + (y-b)² = r₂² -/
  c2 : ℝ → ℝ → Prop

/-- Definition of separate circles -/
def are_separate (circles : TwoCircles) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ r₁ r₂ : ℝ),
    (∀ x y, circles.c1 x y ↔ (x - x₁)^2 + (y - y₁)^2 = r₁^2) ∧
    (∀ x y, circles.c2 x y ↔ (x - x₂)^2 + (y - y₂)^2 = r₂^2) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 > (r₁ + r₂)^2

/-- The two given circles are separate -/
theorem circles_are_separate : are_separate { 
  c1 := λ x y => x^2 + y^2 = 1,
  c2 := λ x y => (x-3)^2 + (y-4)^2 = 9
} := by sorry

end circles_are_separate_l3559_355994


namespace recipe_flour_amount_l3559_355977

/-- A baking recipe with flour and sugar -/
structure Recipe where
  flour : ℕ
  sugar : ℕ

/-- The amount of flour Mary has already added -/
def flour_added : ℕ := 2

/-- The amount of flour Mary still needs to add -/
def flour_to_add : ℕ := 7

/-- The recipe Mary is using -/
def marys_recipe : Recipe := {
  flour := flour_added + flour_to_add,
  sugar := 3
}

/-- Theorem: The total amount of flour in the recipe is equal to the sum of the flour already added and the flour to be added -/
theorem recipe_flour_amount : marys_recipe.flour = flour_added + flour_to_add := by
  sorry

end recipe_flour_amount_l3559_355977


namespace probability_is_one_twelfth_l3559_355935

/-- Represents the outcome of rolling two 6-sided dice -/
def DiceRoll := Fin 6 × Fin 6

/-- Calculates the sum of a dice roll -/
def sum_roll (roll : DiceRoll) : Nat :=
  (roll.1.val + 1) + (roll.2.val + 1)

/-- Represents the sample space of all possible dice rolls -/
def sample_space : Finset DiceRoll :=
  Finset.product (Finset.univ : Finset (Fin 6)) (Finset.univ : Finset (Fin 6))

/-- Checks if the area of a circle is less than its circumference given its diameter -/
def area_less_than_circumference (d : Nat) : Bool :=
  d * d < 4 * d

/-- The set of favorable outcomes -/
def favorable_outcomes : Finset DiceRoll :=
  sample_space.filter (λ roll => area_less_than_circumference (sum_roll roll))

/-- The probability of the area being less than the circumference -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem probability_is_one_twelfth : probability = 1 / 12 := by
  sorry

end probability_is_one_twelfth_l3559_355935


namespace inequality_solution_implies_m_value_l3559_355980

theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, mx + 2 > 0 ↔ x < 2) → m = -1 := by
  sorry

end inequality_solution_implies_m_value_l3559_355980


namespace charlie_has_largest_answer_l3559_355992

def alice_calc (start : ℕ) : ℕ := ((start - 3) * 3) + 5

def bob_calc (start : ℕ) : ℕ := ((start * 3) - 3) + 5

def charlie_calc (start : ℕ) : ℕ := ((start - 3) + 5) * 3

theorem charlie_has_largest_answer (start : ℕ) (h : start = 15) :
  charlie_calc start > alice_calc start ∧ charlie_calc start > bob_calc start := by
  sorry

end charlie_has_largest_answer_l3559_355992


namespace max_sum_roots_l3559_355941

/-- Given real numbers b and c, and function f(x) = x^2 + bx + c,
    if f(f(x)) = 0 has exactly three different real roots,
    then the maximum value of the sum of the roots of f(x) is 1/2. -/
theorem max_sum_roots (b c : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0) →
  (∃ (α β : ℝ), f α = 0 ∧ f β = 0 ∧ α + β = -b) →
  (∀ (x y : ℝ), f x = 0 ∧ f y = 0 → x + y ≤ 1/2) :=
by sorry

end max_sum_roots_l3559_355941


namespace distribution_theorem_l3559_355919

/-- The number of ways to distribute 5 students into 3 groups (A, B, C),
    where group A has at least 2 students and groups B and C each have at least 1 student. -/
def distribution_schemes : ℕ := 80

/-- The total number of students -/
def total_students : ℕ := 5

/-- The number of groups -/
def num_groups : ℕ := 3

/-- The minimum number of students in group A -/
def min_group_a : ℕ := 2

/-- The minimum number of students in groups B and C -/
def min_group_bc : ℕ := 1

theorem distribution_theorem :
  (∀ (scheme : Fin total_students → Fin num_groups),
    (∃ (a b c : Finset (Fin total_students)),
      a.card ≥ min_group_a ∧
      b.card ≥ min_group_bc ∧
      c.card ≥ min_group_bc ∧
      a ∪ b ∪ c = Finset.univ ∧
      a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅)) →
  (Fintype.card {scheme : Fin total_students → Fin num_groups |
    ∃ (a b c : Finset (Fin total_students)),
      a.card ≥ min_group_a ∧
      b.card ≥ min_group_bc ∧
      c.card ≥ min_group_bc ∧
      a ∪ b ∪ c = Finset.univ ∧
      a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅}) = distribution_schemes :=
by sorry

end distribution_theorem_l3559_355919


namespace complex_solutions_of_x_squared_equals_negative_four_l3559_355943

theorem complex_solutions_of_x_squared_equals_negative_four :
  ∀ x : ℂ, x^2 = -4 ↔ x = 2*I ∨ x = -2*I :=
sorry

end complex_solutions_of_x_squared_equals_negative_four_l3559_355943


namespace zero_points_product_bound_l3559_355947

noncomputable def f (a x : ℝ) : ℝ := |Real.log x / Real.log a| - (1/2)^x

theorem zero_points_product_bound (a x₁ x₂ : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) 
  (hx₁ : f a x₁ = 0) 
  (hx₂ : f a x₂ = 0) : 
  0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry

end zero_points_product_bound_l3559_355947


namespace parallelogram_area_parallelogram_area_proof_l3559_355998

/-- A parallelogram with vertices at (0, 0), (7, 0), (3, 5), and (10, 5) has an area of 35 square units. -/
theorem parallelogram_area : ℝ → Prop := fun area =>
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (7, 0)
  let v3 : ℝ × ℝ := (3, 5)
  let v4 : ℝ × ℝ := (10, 5)
  area = 35

/-- The proof of the parallelogram area theorem. -/
theorem parallelogram_area_proof : parallelogram_area 35 := by
  sorry

end parallelogram_area_parallelogram_area_proof_l3559_355998


namespace car_trip_duration_l3559_355920

theorem car_trip_duration : ∀ (x : ℝ),
  let d1 : ℝ := 70 * 4  -- distance covered in first segment
  let d2 : ℝ := 60 * 5  -- distance covered in second segment
  let d3 : ℝ := 50 * x  -- distance covered in third segment
  let total_distance : ℝ := d1 + d2 + d3
  let total_time : ℝ := 4 + 5 + x
  let average_speed : ℝ := 58
  average_speed = total_distance / total_time →
  total_time = 16.25 :=
by sorry


end car_trip_duration_l3559_355920


namespace matt_profit_l3559_355937

/-- Represents a baseball card collection --/
structure CardCollection where
  count : ℕ
  value : ℕ

/-- Calculates the total value of a card collection --/
def totalValue (c : CardCollection) : ℕ := c.count * c.value

/-- Represents a trade transaction --/
structure Trade where
  givenCards : List CardCollection
  receivedCards : List CardCollection

/-- Calculates the profit from a trade --/
def tradeProfitᵢ (t : Trade) : ℤ :=
  (t.receivedCards.map totalValue).sum - (t.givenCards.map totalValue).sum

/-- The initial card collection --/
def initialCollection : CardCollection := ⟨8, 6⟩

/-- The four trades Matt made --/
def trades : List Trade := [
  ⟨[⟨2, 6⟩], [⟨3, 2⟩, ⟨1, 9⟩]⟩,
  ⟨[⟨1, 2⟩, ⟨1, 6⟩], [⟨2, 5⟩, ⟨1, 8⟩]⟩,
  ⟨[⟨1, 5⟩, ⟨1, 9⟩], [⟨3, 3⟩, ⟨1, 10⟩, ⟨1, 1⟩]⟩,
  ⟨[⟨2, 3⟩, ⟨1, 8⟩], [⟨2, 7⟩, ⟨1, 4⟩]⟩
]

/-- Calculates the total profit from all trades --/
def totalProfit : ℤ := (trades.map tradeProfitᵢ).sum

theorem matt_profit : totalProfit = 23 := by
  sorry

end matt_profit_l3559_355937


namespace complex_square_root_l3559_355923

theorem complex_square_root (z : ℂ) : 
  z^2 = -3 - 4*I ∧ z.re < 0 ∧ z.im > 0 → z = -1 + 2*I :=
by sorry

end complex_square_root_l3559_355923


namespace ceiling_fraction_evaluation_l3559_355949

theorem ceiling_fraction_evaluation :
  (⌈(25 / 11 : ℚ) - ⌈(35 / 25 : ℚ)⌉⌉ : ℚ) /
  (⌈(35 / 11 : ℚ) + ⌈(11 * 25 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 12 := by
  sorry

end ceiling_fraction_evaluation_l3559_355949


namespace worker_selection_theorem_l3559_355953

/-- The number of workers who can only work as pliers workers -/
def pliers_only : ℕ := 5

/-- The number of workers who can only work as car workers -/
def car_only : ℕ := 4

/-- The number of workers who can work both as pliers and car workers -/
def both : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := pliers_only + car_only + both

/-- The number of workers to be selected for pliers -/
def pliers_needed : ℕ := 4

/-- The number of workers to be selected for cars -/
def cars_needed : ℕ := 4

/-- The function to calculate the number of ways to select workers -/
def select_workers : ℕ := sorry

theorem worker_selection_theorem : select_workers = 185 := by sorry

end worker_selection_theorem_l3559_355953


namespace complex_equation_solution_l3559_355942

theorem complex_equation_solution (z : ℂ) : (2 * I) / z = 1 - I → z = -1 + I := by
  sorry

end complex_equation_solution_l3559_355942


namespace sum_of_cubes_divisibility_l3559_355981

theorem sum_of_cubes_divisibility (n : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3 * n * (n^2 + 2) = 3 * k₁ ∧ 3 * n * (n^2 + 2) = 9 * k₂ := by
  sorry

end sum_of_cubes_divisibility_l3559_355981


namespace new_person_weight_l3559_355983

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 65 ∧
  avg_increase = 4 →
  n * avg_increase + replaced_weight = 97 :=
by sorry

end new_person_weight_l3559_355983


namespace danny_bottle_caps_l3559_355928

/-- Represents the number of bottle caps in various situations -/
structure BottleCaps where
  found : ℕ
  thrown_away : ℕ
  current : ℕ

/-- Given Danny's bottle cap collection data, prove that he found 1 more than he threw away -/
theorem danny_bottle_caps (caps : BottleCaps)
  (h1 : caps.found = 36)
  (h2 : caps.thrown_away = 35)
  (h3 : caps.current = 22) :
  caps.found - caps.thrown_away = 1 := by
  sorry

end danny_bottle_caps_l3559_355928


namespace ball_probability_problem_l3559_355961

theorem ball_probability_problem (R B : ℕ) : 
  (R * (R - 1)) / ((R + B) * (R + B - 1)) = 2/7 →
  (2 * R * B) / ((R + B) * (R + B - 1)) = 1/2 →
  R = 105 ∧ B = 91 := by
sorry

end ball_probability_problem_l3559_355961


namespace fundraising_theorem_l3559_355931

def fundraising_problem (goal ken_amount : ℕ) : Prop :=
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  let total_raised := ken_amount + mary_amount + scott_amount
  (mary_amount = 5 * ken_amount) ∧
  (mary_amount = 3 * scott_amount) ∧
  (ken_amount = 600) ∧
  (total_raised - goal = 600)

theorem fundraising_theorem : fundraising_problem 4000 600 := by
  sorry

end fundraising_theorem_l3559_355931


namespace final_amount_is_correct_l3559_355969

/-- Calculates the final amount paid for a shopping trip with specific discounts and promotions -/
def calculate_final_amount (jimmy_shorts : ℕ) (jimmy_short_price : ℚ) 
                           (irene_shirts : ℕ) (irene_shirt_price : ℚ) 
                           (senior_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let jimmy_total := jimmy_shorts * jimmy_short_price
  let irene_total := irene_shirts * irene_shirt_price
  let jimmy_discounted := (jimmy_shorts / 3) * 2 * jimmy_short_price
  let irene_discounted := ((irene_shirts / 3) * 2 + irene_shirts % 3) * irene_shirt_price
  let total_before_discount := jimmy_discounted + irene_discounted
  let discount_amount := total_before_discount * senior_discount
  let total_after_discount := total_before_discount - discount_amount
  let tax_amount := total_after_discount * sales_tax
  total_after_discount + tax_amount

/-- Theorem stating that the final amount paid is $76.55 -/
theorem final_amount_is_correct : 
  calculate_final_amount 3 15 5 17 (1/10) (1/20) = 76.55 := by
  sorry

end final_amount_is_correct_l3559_355969


namespace variance_scaling_l3559_355901

def variance (data : List ℝ) : ℝ := sorry

theorem variance_scaling (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  let data₁ := [a₁, a₂, a₃, a₄, a₅, a₆]
  let data₂ := [2*a₁, 2*a₂, 2*a₃, 2*a₄, 2*a₅, 2*a₆]
  variance data₁ = 2 → variance data₂ = 8 := by
  sorry

end variance_scaling_l3559_355901


namespace price_reduction_equation_l3559_355936

/-- Represents the price reduction scenario for a certain type of chip -/
theorem price_reduction_equation (initial_price : ℝ) (final_price : ℝ) (x : ℝ) :
  initial_price = 400 →
  final_price = 144 →
  0 < x →
  x < 1 →
  initial_price * (1 - x)^2 = final_price :=
by sorry

end price_reduction_equation_l3559_355936


namespace hiker_cyclist_catchup_time_l3559_355916

/-- Proves that a hiker catches up to a cyclist in 10 minutes under specific conditions -/
theorem hiker_cyclist_catchup_time :
  let hiker_speed : ℝ := 4  -- km/h
  let cyclist_speed : ℝ := 12  -- km/h
  let stop_time : ℝ := 5 / 60  -- hours (5 minutes converted to hours)
  
  let distance_cyclist : ℝ := cyclist_speed * stop_time
  let distance_hiker : ℝ := hiker_speed * stop_time
  let distance_between : ℝ := distance_cyclist - distance_hiker
  
  let catchup_time : ℝ := distance_between / hiker_speed

  catchup_time * 60 = 10  -- Convert hours to minutes
  := by sorry

end hiker_cyclist_catchup_time_l3559_355916


namespace unique_number_satisfying_equation_l3559_355915

theorem unique_number_satisfying_equation : ∃! x : ℝ, ((x^3)^(1/3) * 4) / 2 + 5 = 15 :=
by
  sorry

end unique_number_satisfying_equation_l3559_355915


namespace calculator_theorem_l3559_355910

/-- Represents the state of the calculator as a 4-tuple of real numbers -/
def CalculatorState := Fin 4 → ℝ

/-- Applies the transformation to a given state -/
def transform (s : CalculatorState) : CalculatorState :=
  fun i => match i with
  | 0 => s 0 - s 1
  | 1 => s 1 - s 2
  | 2 => s 2 - s 3
  | 3 => s 3 - s 0

/-- Applies the transformation n times to a given state -/
def transformN (s : CalculatorState) (n : ℕ) : CalculatorState :=
  match n with
  | 0 => s
  | n + 1 => transform (transformN s n)

/-- Checks if any number in the state is greater than 1985 -/
def hasLargeNumber (s : CalculatorState) : Prop :=
  ∃ i : Fin 4, s i > 1985

/-- Main theorem statement -/
theorem calculator_theorem (s : CalculatorState) 
  (h : ∃ i j : Fin 4, s i ≠ s j) : 
  ∃ n : ℕ, hasLargeNumber (transformN s n) := by
sorry

end calculator_theorem_l3559_355910


namespace xy_value_l3559_355940

theorem xy_value (x y : ℂ) (h : (1 - Complex.I) * x + (1 + Complex.I) * y = 2) : x * y = 1 := by
  sorry

end xy_value_l3559_355940


namespace youngest_sibling_age_l3559_355904

theorem youngest_sibling_age (a b c d : ℕ) : 
  a + b + c + d = 180 →
  b = a + 2 →
  c = a + 4 →
  d = a + 6 →
  Even a →
  Even b →
  Even c →
  Even d →
  a = 42 := by
sorry

end youngest_sibling_age_l3559_355904


namespace ice_cream_sundaes_l3559_355962

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  n + n.choose 2 = 36 := by
  sorry

end ice_cream_sundaes_l3559_355962


namespace b_n_formula_l3559_355956

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem b_n_formula (a b : ℕ → ℚ) :
  arithmetic_sequence a →
  a 3 = 2 →
  a 8 = 12 →
  b 1 = 4 →
  (∀ n : ℕ, n > 1 → a n + b n = b (n - 1)) →
  ∀ n : ℕ, b n = -n^2 + 3*n + 2 := by
sorry

end b_n_formula_l3559_355956


namespace prob_same_color_top_three_l3559_355918

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of cards of each color (red or black) in a standard deck -/
def cardsPerColor : ℕ := standardDeckSize / 2

/-- The probability of drawing three cards of the same color from the top of a randomly arranged standard deck -/
def probSameColorTopThree : ℚ :=
  (2 * cardsPerColor * (cardsPerColor - 1) * (cardsPerColor - 2)) /
  (standardDeckSize * (standardDeckSize - 1) * (standardDeckSize - 2))

theorem prob_same_color_top_three :
  probSameColorTopThree = 1 / 17 := by
  sorry

end prob_same_color_top_three_l3559_355918


namespace average_water_added_l3559_355972

def water_day1 : ℝ := 318
def water_day2 : ℝ := 312
def water_day3_morning : ℝ := 180
def water_day3_afternoon : ℝ := 162
def num_days : ℝ := 3

theorem average_water_added (water_day1 water_day2 water_day3_morning water_day3_afternoon num_days : ℝ) :
  (water_day1 + water_day2 + water_day3_morning + water_day3_afternoon) / num_days = 324 := by
  sorry

end average_water_added_l3559_355972


namespace length_of_chord_line_equation_l3559_355945

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (1,0)
def line_through_P (m b : ℝ) (x y : ℝ) : Prop := y = m*(x - 1)

-- Define the intersection points of the line and parabola
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | parabola x y ∧ line_through_P m b x y}

-- Part 1: Length of chord AB when slope is 1
theorem length_of_chord (A B : ℝ × ℝ) :
  A ∈ intersection_points 1 0 →
  B ∈ intersection_points 1 0 →
  A ≠ B →
  ‖A - B‖ = 2 * Real.sqrt 6 :=
sorry

-- Part 2: Equation of line when PA = -2PB
theorem line_equation (A B : ℝ × ℝ) (m : ℝ) :
  A ∈ intersection_points m 0 →
  B ∈ intersection_points m 0 →
  A ≠ B →
  (A.1 - 1, A.2) = (-2 * (B.1 - 1), -2 * B.2) →
  (m = 1/2 ∨ m = -1/2) :=
sorry

end length_of_chord_line_equation_l3559_355945


namespace square_completion_and_max_value_l3559_355900

theorem square_completion_and_max_value :
  -- Part 1: Completing the square
  ∀ a : ℝ, a^2 + 4*a + 4 = (a + 2)^2 ∧
  -- Part 2: Factorization using completion of squares
  ∀ a : ℝ, a^2 - 24*a + 143 = (a - 11)*(a - 13) ∧
  -- Part 3: Maximum value of quadratic function
  ∀ a : ℝ, -1/4*a^2 + 2*a - 1 ≤ 3 :=
by
  sorry

end square_completion_and_max_value_l3559_355900


namespace only_B_is_equation_l3559_355973

-- Define what an equation is
def is_equation (e : String) : Prop :=
  ∃ (lhs rhs : String), e = lhs ++ "=" ++ rhs

-- Define the given expressions
def expr_A : String := "x-6"
def expr_B : String := "3r+y=5"
def expr_C : String := "-3+x>-2"
def expr_D : String := "4/6=2/3"

-- Theorem statement
theorem only_B_is_equation :
  is_equation expr_B ∧
  ¬is_equation expr_A ∧
  ¬is_equation expr_C ∧
  ¬is_equation expr_D :=
by sorry

end only_B_is_equation_l3559_355973


namespace infinite_solutions_l3559_355933

-- Define α as the positive root of x^2 - 1989x - 1 = 0
noncomputable def α : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

-- Define the equation we want to prove holds for infinitely many n
def equation (n : ℕ) : Prop :=
  ⌊α * n + 1989 * α * ⌊α * n⌋⌋ = 1989 * n + (1989^2 + 1) * ⌊α * n⌋

-- Theorem statement
theorem infinite_solutions :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → equation n :=
sorry

end infinite_solutions_l3559_355933


namespace aluminum_sulfide_weight_l3559_355930

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight (al_weight s_weight : ℝ) : ℝ :=
  2 * al_weight + 3 * s_weight

/-- The weight of a given number of moles of a compound -/
def molar_weight (moles molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

theorem aluminum_sulfide_weight :
  let al_weight : ℝ := 26.98
  let s_weight : ℝ := 32.06
  let moles : ℝ := 4
  molar_weight moles (molecular_weight al_weight s_weight) = 600.56 := by
sorry

end aluminum_sulfide_weight_l3559_355930


namespace largest_non_prime_consecutive_l3559_355982

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + i)

theorem largest_non_prime_consecutive :
  ∃ (start : ℕ),
    start + 5 = 35 ∧
    start < 50 ∧
    (∀ n ∈ consecutive_integers start 6, n < 50 ∧ ¬(is_prime n)) ∧
    (∀ m : ℕ, m > start + 5 →
      ¬(∃ s : ℕ, s + 5 = m ∧
        s < 50 ∧
        (∀ n ∈ consecutive_integers s 6, n < 50 ∧ ¬(is_prime n)))) :=
sorry

end largest_non_prime_consecutive_l3559_355982


namespace distance_sum_theorem_l3559_355967

theorem distance_sum_theorem (x z w : ℝ) 
  (hx : x = -1)
  (hz : z = 3.7)
  (hw : w = 9.3) :
  |z - x| + |w - x| = 15 := by
sorry

end distance_sum_theorem_l3559_355967


namespace sum_of_A_and_C_l3559_355914

theorem sum_of_A_and_C (A B C : ℕ) : A = 238 → A = B + 143 → C = B + 304 → A + C = 637 := by
  sorry

end sum_of_A_and_C_l3559_355914


namespace minimum_fencing_cost_theorem_l3559_355951

/-- Represents the cost per linear foot for different fencing materials -/
structure FencingMaterial where
  wood : ℝ
  chainLink : ℝ
  iron : ℝ

/-- Calculates the minimum fencing cost for a rectangular field -/
def minimumFencingCost (area : ℝ) (uncoveredSide : ℝ) (materials : FencingMaterial) : ℝ :=
  sorry

/-- Theorem stating the minimum fencing cost for the given problem -/
theorem minimum_fencing_cost_theorem :
  let area : ℝ := 680
  let uncoveredSide : ℝ := 34
  let materials : FencingMaterial := { wood := 5, chainLink := 7, iron := 10 }
  minimumFencingCost area uncoveredSide materials = 438 := by
  sorry

end minimum_fencing_cost_theorem_l3559_355951


namespace area_constants_sum_l3559_355990

/-- Represents a grid with squares and overlapping circles -/
structure GridWithCircles where
  grid_size : Nat
  square_size : ℝ
  circle_diameter : ℝ
  circle_center_distance : ℝ

/-- Calculates the constants C and D for the area of visible shaded region -/
def calculate_area_constants (g : GridWithCircles) : ℝ × ℝ :=
  sorry

/-- The theorem stating that C + D = 150 for the given configuration -/
theorem area_constants_sum (g : GridWithCircles) 
  (h1 : g.grid_size = 4)
  (h2 : g.square_size = 3)
  (h3 : g.circle_diameter = 6)
  (h4 : g.circle_center_distance = 3) :
  let (C, D) := calculate_area_constants g
  C + D = 150 :=
sorry

end area_constants_sum_l3559_355990


namespace problem_statement_l3559_355958

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a-b, 0} → a^2019 + b^2019 = -1 := by
  sorry

end problem_statement_l3559_355958


namespace train_crossing_problem_l3559_355965

/-- Calculates the length of a train given the length and speed of another train, 
    their crossing time, and the speed of the train we're calculating. -/
def train_length (other_length : ℝ) (other_speed : ℝ) (this_speed : ℝ) (cross_time : ℝ) : ℝ :=
  ((other_speed + this_speed) * cross_time - other_length)

theorem train_crossing_problem : 
  let first_train_length : ℝ := 290
  let first_train_speed : ℝ := 120 * 1000 / 3600  -- Convert km/h to m/s
  let second_train_speed : ℝ := 80 * 1000 / 3600  -- Convert km/h to m/s
  let crossing_time : ℝ := 9
  abs (train_length first_train_length first_train_speed second_train_speed crossing_time - 209.95) < 0.01 := by
  sorry

end train_crossing_problem_l3559_355965


namespace solution_set_and_range_l3559_355960

def f (a x : ℝ) : ℝ := -x^2 + a*x + 4

def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem solution_set_and_range :
  (∀ x ∈ Set.Icc (-1 : ℝ) ((Real.sqrt 17 - 1) / 2), f 1 x ≥ g x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x > g x) ∧
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ g x) ∧
  (∀ a < -1, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x) ∧
  (∀ a > 1, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x) :=
by sorry

end solution_set_and_range_l3559_355960


namespace simplify_sqrt_product_l3559_355917

theorem simplify_sqrt_product : 
  Real.sqrt (5 * 3) * Real.sqrt (3^5 * 5^2) = 135 * Real.sqrt 5 := by
  sorry

end simplify_sqrt_product_l3559_355917


namespace market_demand_growth_rate_bound_l3559_355903

theorem market_demand_growth_rate_bound
  (a : Fin 4 → ℝ)  -- Market demand sequence for 4 years
  (p₁ p₂ p₃ : ℝ)   -- Percentage increases between consecutive years
  (h₁ : p₁ + p₂ + p₃ = 1)  -- Condition on percentage increases
  (h₂ : ∀ i : Fin 3, a (i + 1) = a i * (1 + [p₁, p₂, p₃].get i))  -- Relation between consecutive demands
  : ∃ p : ℝ, (∀ i : Fin 3, a (i + 1) = a i * (1 + p)) ∧ p ≤ 1/3 :=
by sorry

end market_demand_growth_rate_bound_l3559_355903


namespace circumcircles_intersect_at_common_point_l3559_355952

-- Define the basic structures
structure Point : Type := (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the properties and conditions
def is_acute_triangle (t : Triangle) : Prop := sorry

def are_not_equal (p q : Point) : Prop := sorry

def is_midpoint (m : Point) (p q : Point) : Prop := sorry

def is_midpoint_of_minor_arc (m : Point) (a b c : Point) : Prop := sorry

def is_midpoint_of_major_arc (n : Point) (a b c : Point) : Prop := sorry

def is_incenter (w : Point) (t : Triangle) : Prop := sorry

def is_excenter (x : Point) (t : Triangle) (v : Point) : Prop := sorry

def circumcircle (t : Triangle) : Circle := sorry

def circles_intersect_at_point (c₁ c₂ c₃ : Circle) (p : Point) : Prop := sorry

-- State the theorem
theorem circumcircles_intersect_at_common_point
  (A B C D E F M N W X Y Z : Point) :
  is_acute_triangle (Triangle.mk A B C) →
  are_not_equal A B →
  are_not_equal A C →
  is_midpoint D B C →
  is_midpoint E C A →
  is_midpoint F A B →
  is_midpoint_of_minor_arc M B C A →
  is_midpoint_of_major_arc N B A C →
  is_incenter W (Triangle.mk D E F) →
  is_excenter X (Triangle.mk D E F) D →
  is_excenter Y (Triangle.mk D E F) E →
  is_excenter Z (Triangle.mk D E F) F →
  ∃ (P : Point),
    circles_intersect_at_point
      (circumcircle (Triangle.mk A B C))
      (circumcircle (Triangle.mk W N X))
      (circumcircle (Triangle.mk Y M Z))
      P :=
by
  sorry

end circumcircles_intersect_at_common_point_l3559_355952


namespace smallest_divisible_by_14_15_16_l3559_355944

theorem smallest_divisible_by_14_15_16 : ∃ n : ℕ, n > 0 ∧ 14 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n ∧ ∀ m : ℕ, m > 0 → 14 ∣ m → 15 ∣ m → 16 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_divisible_by_14_15_16_l3559_355944


namespace cat_food_cans_per_package_l3559_355991

theorem cat_food_cans_per_package (cat_packages : ℕ) (dog_packages : ℕ) (dog_cans_per_package : ℕ) (cat_dog_difference : ℕ) :
  cat_packages = 9 →
  dog_packages = 7 →
  dog_cans_per_package = 5 →
  cat_dog_difference = 55 →
  ∃ (cat_cans_per_package : ℕ),
    cat_cans_per_package * cat_packages = dog_cans_per_package * dog_packages + cat_dog_difference ∧
    cat_cans_per_package = 10 :=
by
  sorry

end cat_food_cans_per_package_l3559_355991


namespace unique_divisible_by_45_l3559_355964

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def five_digit_number (x : ℕ) : ℕ := x * 10000 + 2 * 1000 + 7 * 100 + x * 10 + 5

theorem unique_divisible_by_45 : 
  ∃! x : ℕ, digit x ∧ is_divisible_by (five_digit_number x) 45 ∧ x = 2 := by sorry

end unique_divisible_by_45_l3559_355964


namespace complex_number_real_part_eq_imaginary_part_l3559_355927

theorem complex_number_real_part_eq_imaginary_part (a : ℝ) : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - 2*i) * (a + i)
  (z.re + z.im = 0) → a = 3 := by
  sorry

end complex_number_real_part_eq_imaginary_part_l3559_355927


namespace fraction_equals_five_l3559_355986

theorem fraction_equals_five (a b : ℕ+) (k : ℕ+) 
  (h : (a.val^2 + b.val^2 : ℚ) / (a.val * b.val - 1) = k.val) : k = 5 := by
  sorry

end fraction_equals_five_l3559_355986
