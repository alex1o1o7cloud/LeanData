import Mathlib

namespace radical_simplification_l2399_239908

theorem radical_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (40 * q) * Real.sqrt (20 * q) * Real.sqrt (10 * q) = 40 * q * Real.sqrt (5 * q) := by
  sorry

end radical_simplification_l2399_239908


namespace cube_root_equation_solution_l2399_239943

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end cube_root_equation_solution_l2399_239943


namespace mike_total_spent_l2399_239907

def trumpet_cost : ℚ := 145.16
def songbook_cost : ℚ := 5.84

theorem mike_total_spent :
  trumpet_cost + songbook_cost = 151 := by sorry

end mike_total_spent_l2399_239907


namespace chad_sandwiches_l2399_239934

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights 5 boxes of crackers last Chad -/
def num_nights : ℕ := 56

/-- The number of sandwiches Chad has each night -/
def sandwiches_per_night : ℕ := 5

theorem chad_sandwiches :
  sandwiches_per_night * crackers_per_sandwich * num_nights =
  num_boxes * sleeves_per_box * crackers_per_sleeve :=
sorry

end chad_sandwiches_l2399_239934


namespace searchlight_probability_l2399_239914

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℚ := 2

/-- The time in seconds for one complete revolution of the searchlight -/
def revolution_time : ℚ := 60 / revolutions_per_minute

/-- The minimum time in seconds a man needs to stay in the dark -/
def min_dark_time : ℚ := 10

/-- The probability of a man staying in the dark for at least the minimum time -/
def dark_probability : ℚ := min_dark_time / revolution_time

theorem searchlight_probability :
  dark_probability = 1 / 3 := by sorry

end searchlight_probability_l2399_239914


namespace fourth_board_score_l2399_239922

/-- Represents a dartboard with its score -/
structure Dartboard :=
  (score : ℕ)

/-- Represents the set of four dartboards -/
def Dartboards := Fin 4 → Dartboard

theorem fourth_board_score (boards : Dartboards) 
  (h1 : boards 0 = ⟨30⟩)
  (h2 : boards 1 = ⟨38⟩)
  (h3 : boards 2 = ⟨41⟩)
  (identical : ∀ (i j : Fin 4), (boards i).score + (boards j).score = 2 * ((boards 0).score + (boards 1).score) / 2) :
  (boards 3).score = 34 := by
  sorry

end fourth_board_score_l2399_239922


namespace square_of_negative_sum_l2399_239960

theorem square_of_negative_sum (x y : ℝ) : (-x - y)^2 = x^2 + 2*x*y + y^2 := by
  sorry

end square_of_negative_sum_l2399_239960


namespace tom_bricks_count_l2399_239992

/-- The number of bricks Tom needs to buy -/
def num_bricks : ℕ := 1000

/-- The cost of a brick at full price -/
def full_price : ℚ := 1/2

/-- The total amount Tom spends -/
def total_spent : ℚ := 375

theorem tom_bricks_count :
  (num_bricks / 2 : ℚ) * (full_price / 2) + (num_bricks / 2 : ℚ) * full_price = total_spent :=
sorry

end tom_bricks_count_l2399_239992


namespace problem_2a_l2399_239925

theorem problem_2a (a b x y : ℝ) 
  (eq1 : a * x + b * y = 7)
  (eq2 : a * x^2 + b * y^2 = 49)
  (eq3 : a * x^3 + b * y^3 = 133)
  (eq4 : a * x^4 + b * y^4 = 406) :
  2014 * (x + y - x * y) - 100 * (a + b) = 6889.33 := by
sorry

end problem_2a_l2399_239925


namespace equation_solution_l2399_239903

theorem equation_solution : ∃ x : ℝ, (2 / (x + 3) = 1) ∧ (x = -1) := by
  sorry

end equation_solution_l2399_239903


namespace min_value_trig_expression_l2399_239957

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  (∀ φ, 0 < φ ∧ φ < π / 2 →
    3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≤ 3 * cos φ + 2 / sin φ + 2 * sqrt 2 * tan φ) ∧
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ = 7 * sqrt 2 / 2 :=
sorry

end min_value_trig_expression_l2399_239957


namespace sum_of_squares_l2399_239986

theorem sum_of_squares (x y z a b c k : ℝ) 
  (h1 : x * y = k * a)
  (h2 : x * z = b)
  (h3 : y * z = c)
  (h4 : k ≠ 0)
  (h5 : x ≠ 0)
  (h6 : y ≠ 0)
  (h7 : z ≠ 0)
  (h8 : a ≠ 0)
  (h9 : b ≠ 0)
  (h10 : c ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) := by
  sorry

end sum_of_squares_l2399_239986


namespace pants_cost_is_250_l2399_239995

/-- The cost of each pair of pants given the total cost of t-shirts and pants, 
    the number of t-shirts and pants, and the cost of each t-shirt. -/
def cost_of_pants (total_cost : ℕ) (num_tshirts : ℕ) (num_pants : ℕ) (tshirt_cost : ℕ) : ℕ :=
  (total_cost - num_tshirts * tshirt_cost) / num_pants

/-- Theorem stating that the cost of each pair of pants is 250 
    given the conditions in the problem. -/
theorem pants_cost_is_250 :
  cost_of_pants 1500 5 4 100 = 250 := by
  sorry

end pants_cost_is_250_l2399_239995


namespace potato_difference_l2399_239915

/-- The number of potato wedges Cynthia makes -/
def x : ℕ := 8 * 13

/-- The number of potatoes used for french fries or potato chips -/
def k : ℕ := (67 - 13) / 2

/-- The number of potato chips Cynthia makes -/
def z : ℕ := 20 * k

/-- The difference between the number of potato chips and potato wedges -/
def d : ℤ := z - x

theorem potato_difference : d = 436 := by
  sorry

end potato_difference_l2399_239915


namespace range_of_f_is_real_l2399_239952

-- Define the function f
def f (x : ℝ) : ℝ := -4 * x + 5

-- Theorem stating that the range of f is ℝ
theorem range_of_f_is_real : Set.range f = Set.univ :=
sorry

end range_of_f_is_real_l2399_239952


namespace largest_angle_in_special_triangle_l2399_239948

/-- Given a triangle with angles in the ratio 3:4:9 and an external angle equal to the smallest 
    internal angle attached at the largest angle, prove that the largest internal angle is 101.25°. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
    b = (4/3) * a ∧ c = 3 * a →  -- Ratio of angles is 3:4:9
    a + b + c = 180 →  -- Sum of internal angles is 180°
    c + a = 12 * a →  -- External angle equals smallest internal angle
    c = 101.25 := by sorry

end largest_angle_in_special_triangle_l2399_239948


namespace cube_divided_by_self_l2399_239935

theorem cube_divided_by_self (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 := by
  sorry

end cube_divided_by_self_l2399_239935


namespace f_divisible_by_8_l2399_239972

def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

theorem f_divisible_by_8 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, f n = 8 * k := by
  sorry

end f_divisible_by_8_l2399_239972


namespace trigonometric_equation_solutions_l2399_239961

theorem trigonometric_equation_solutions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 0 (Real.pi / 2) ∧ y ∈ Set.Icc 0 (Real.pi / 2) ∧ 
   Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x) = a + 1 ∧
   Real.cos (2 * y) + Real.sqrt 3 * Real.sin (2 * y) = a + 1) →
  0 ≤ a ∧ a < 1 :=
by sorry

end trigonometric_equation_solutions_l2399_239961


namespace union_of_A_and_B_l2399_239924

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} := by sorry

end union_of_A_and_B_l2399_239924


namespace inequality_preservation_l2399_239917

theorem inequality_preservation (m n : ℝ) (h : m > n) : (1/5) * m > (1/5) * n := by
  sorry

end inequality_preservation_l2399_239917


namespace one_switch_determines_light_l2399_239955

/-- Represents the state of a switch -/
inductive SwitchState
| Position1
| Position2
| Position3

/-- Represents a light bulb -/
inductive Light
| Bulb1
| Bulb2
| Bulb3

/-- Configuration of all switches -/
def SwitchConfig (n : ℕ) := Fin n → SwitchState

/-- Function that determines which light is on given a switch configuration -/
def lightOn (n : ℕ) (config : SwitchConfig n) : Light := sorry

theorem one_switch_determines_light (n : ℕ) :
  (∀ (config : SwitchConfig n), ∃! (l : Light), lightOn n config = l) →
  (∀ (config1 config2 : SwitchConfig n), 
    (∀ i, config1 i ≠ config2 i) → lightOn n config1 ≠ lightOn n config2) →
  ∃ (k : Fin n), ∀ (config1 config2 : SwitchConfig n),
    (∀ (i : Fin n), i ≠ k → config1 i = config2 i) →
    (config1 k = config2 k → lightOn n config1 = lightOn n config2) :=
sorry

end one_switch_determines_light_l2399_239955


namespace mary_fruit_cost_l2399_239967

/-- Calculates the total cost of fruits with a discount applied -/
def fruitCost (applePrice orangePrice bananaPrice : ℚ) 
              (appleCount orangeCount bananaCount : ℕ) 
              (fruitPerDiscount : ℕ) (discountAmount : ℚ) : ℚ :=
  let totalFruits := appleCount + orangeCount + bananaCount
  let subtotal := applePrice * appleCount + orangePrice * orangeCount + bananaPrice * bananaCount
  let discountCount := totalFruits / fruitPerDiscount
  subtotal - (discountCount * discountAmount)

/-- Theorem stating that Mary will pay $15 for her fruits -/
theorem mary_fruit_cost : 
  fruitCost 1 2 3 5 3 2 5 1 = 15 := by
  sorry

end mary_fruit_cost_l2399_239967


namespace liquid_film_radius_l2399_239940

theorem liquid_film_radius (volume : ℝ) (thickness : ℝ) (radius : ℝ) : 
  volume = 320 →
  thickness = 0.05 →
  volume = π * radius^2 * thickness →
  radius = Real.sqrt (6400 / π) := by
sorry

end liquid_film_radius_l2399_239940


namespace rotated_solid_properties_l2399_239968

/-- A right-angled triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angled : a^2 + b^2 = c^2
  side_a : a = 3
  side_b : b = 4
  side_c : c = 5

/-- The solid formed by rotating the triangle around its hypotenuse -/
def RotatedSolid (t : RightTriangle) : Prop :=
  ∃ (surface_area volume : ℝ),
    surface_area = 84/5 * Real.pi ∧
    volume = 48/5 * Real.pi

/-- Theorem stating the surface area and volume of the rotated solid -/
theorem rotated_solid_properties (t : RightTriangle) :
  RotatedSolid t := by sorry

end rotated_solid_properties_l2399_239968


namespace mean_of_sequence_mean_of_sequence_is_17_75_l2399_239937

theorem mean_of_sequence : Real → Prop :=
  fun mean =>
    let sequence := [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2]
    mean = (sequence.sum : Real) / sequence.length ∧ mean = 17.75

-- The proof is omitted
theorem mean_of_sequence_is_17_75 : ∃ mean, mean_of_sequence mean :=
  sorry

end mean_of_sequence_mean_of_sequence_is_17_75_l2399_239937


namespace arithmetic_geometric_ratio_l2399_239966

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  ∃ q : ℝ, y = x * q ∧ z = y * q

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1 + 2) (a 5 + 5) (a 9 + 8) →
  ∃ q : ℝ, geometric_sequence (a 1 + 2) (a 5 + 5) (a 9 + 8) ∧ q = 1 := by
  sorry

end arithmetic_geometric_ratio_l2399_239966


namespace quadratic_decreasing_threshold_l2399_239981

/-- Represents a quadratic function of the form ax^2 - 2ax + 1 -/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

/-- Proves that for a quadratic function f(x) = ax^2 - 2ax + 1 where a < 0,
    the minimum value of m for which f(x) is decreasing for all x > m is 1 -/
theorem quadratic_decreasing_threshold (a : ℝ) (h : a < 0) :
  ∃ m : ℝ, m = 1 ∧ ∀ x > m, ∀ y > x,
    QuadraticFunction a y < QuadraticFunction a x :=
by sorry

end quadratic_decreasing_threshold_l2399_239981


namespace consecutive_integers_sqrt_17_l2399_239932

theorem consecutive_integers_sqrt_17 (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ b = a + 1 ∧ (a : ℝ) < Real.sqrt 17 ∧ Real.sqrt 17 < (b : ℝ) → a + b = 9 := by
  sorry

end consecutive_integers_sqrt_17_l2399_239932


namespace tetrahedron_rotation_common_volume_l2399_239920

theorem tetrahedron_rotation_common_volume
  (V : ℝ) (α : ℝ) (h : 0 < α ∧ α < π) :
  ∃ (common_volume : ℝ),
    common_volume = V * (1 + Real.tan (α/2)^2) / (1 + Real.tan (α/2))^2 :=
by sorry

end tetrahedron_rotation_common_volume_l2399_239920


namespace jelly_bean_probability_l2399_239990

theorem jelly_bean_probability (red green yellow blue : ℕ) 
  (h_red : red = 7)
  (h_green : green = 9)
  (h_yellow : yellow = 4)
  (h_blue : blue = 10) :
  (red : ℚ) / (red + green + yellow + blue) = 7 / 30 := by
sorry

end jelly_bean_probability_l2399_239990


namespace sum_of_odds_l2399_239919

theorem sum_of_odds (sum_of_evens : ℕ) (n : ℕ) :
  (n = 70) →
  (sum_of_evens = n / 2 * (2 + n * 2)) →
  (sum_of_evens = 4970) →
  (n / 2 * (1 + (n * 2 - 1)) = 4900) :=
by sorry

end sum_of_odds_l2399_239919


namespace dog_food_theorem_l2399_239994

/-- The number of cups of dog food in a bag that lasts 16 days -/
def cups_in_bag (morning_cups : ℕ) (evening_cups : ℕ) (days : ℕ) : ℕ :=
  (morning_cups + evening_cups) * days

/-- Theorem stating that a bag lasting 16 days contains 32 cups of dog food -/
theorem dog_food_theorem :
  cups_in_bag 1 1 16 = 32 := by
  sorry

end dog_food_theorem_l2399_239994


namespace inverse_variation_problem_l2399_239900

/-- Given that a² varies inversely with b², prove that a² = 25/16 when b = 8, given a = 5 when b = 2 -/
theorem inverse_variation_problem (a b : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, x^2 * y^2 = k) 
  (h1 : 5^2 * 2^2 = a^2 * b^2) : 
  8^2 * (25/16) = a^2 * 8^2 := by sorry

end inverse_variation_problem_l2399_239900


namespace point_constraints_l2399_239958

theorem point_constraints (x y : ℝ) :
  x^2 + y^2 ≤ 2 →
  -1 ≤ x / (x + y) →
  x / (x + y) ≤ 1 →
  0 ≤ y ∧ -2*x ≤ y :=
by sorry

end point_constraints_l2399_239958


namespace ellipse_axis_endpoints_distance_l2399_239970

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x - 3)^2 + 16 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the lengths of semi-major and semi-minor axes
def a : ℝ := 4
def b : ℝ := 2

-- Define an endpoint of the major axis
def C : ℝ × ℝ := (center.1 + a, center.2)

-- Define an endpoint of the minor axis
def D : ℝ × ℝ := (center.1, center.2 + b)

-- Theorem statement
theorem ellipse_axis_endpoints_distance : 
  let distance := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  distance = 2 * Real.sqrt 5 := by sorry

end ellipse_axis_endpoints_distance_l2399_239970


namespace triangle_has_inside_altitude_l2399_239928

-- Define a triangle
def Triangle : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define an altitude of a triangle
def Altitude (t : Triangle) : Type := ℝ × ℝ × ℝ × ℝ

-- Define what it means for an altitude to be inside a triangle
def IsInside (a : Altitude t) (t : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_has_inside_altitude (t : Triangle) : 
  ∃ (a : Altitude t), IsInside a t := sorry

end triangle_has_inside_altitude_l2399_239928


namespace triangle_abc_proof_l2399_239989

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  -- Given condition
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0 →
  -- Additional conditions
  a = 2 →
  1/2 * b * c * Real.sin A = Real.sqrt 3 →
  -- Conclusion
  A = π/3 ∧ b = 2 ∧ c = 2 := by
sorry

end triangle_abc_proof_l2399_239989


namespace six_by_six_tiling_impossible_l2399_239996

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a tile -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Represents a tiling configuration -/
structure TilingConfig :=
  (board : Chessboard)
  (tile : Tile)
  (num_tiles : Nat)

/-- Predicate to check if a tiling configuration is valid -/
def is_valid_tiling (config : TilingConfig) : Prop :=
  config.board.rows * config.board.cols = config.tile.length * config.tile.width * config.num_tiles

/-- Theorem stating that a 6x6 chessboard cannot be tiled with nine 1x4 tiles -/
theorem six_by_six_tiling_impossible :
  ¬ is_valid_tiling { board := { rows := 6, cols := 6 },
                      tile := { length := 1, width := 4 },
                      num_tiles := 9 } :=
by sorry

end six_by_six_tiling_impossible_l2399_239996


namespace quadratic_equation_problem_l2399_239956

theorem quadratic_equation_problem (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + 2*m*x₁ + m^2 - m + 2 = 0 ∧
    x₂^2 + 2*m*x₂ + m^2 - m + 2 = 0 ∧
    x₁ + x₂ + x₁ * x₂ = 2) →
  m = 3 := by
sorry

end quadratic_equation_problem_l2399_239956


namespace inserted_sequence_theorem_l2399_239945

/-- Given a sequence, insert_between inserts n elements between each pair of adjacent elements -/
def insert_between (seq : ℕ → α) (n : ℕ) : ℕ → α :=
  λ k => if k % (n + 1) = 0 then seq (k / (n + 1) + 1) else seq (k / (n + 1) + 1)

theorem inserted_sequence_theorem (original_seq : ℕ → α) :
  (insert_between original_seq 3) 69 = original_seq 18 := by
  sorry

end inserted_sequence_theorem_l2399_239945


namespace empty_solution_implies_a_leq_5_l2399_239910

theorem empty_solution_implies_a_leq_5 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 2| + |x + 3| < a)) → a ≤ 5 := by
  sorry

end empty_solution_implies_a_leq_5_l2399_239910


namespace sum_of_three_digit_permutations_not_2018_l2399_239980

theorem sum_of_three_digit_permutations_not_2018 (a b c : ℕ) : 
  (0 < a ∧ a ≤ 9) → (0 < b ∧ b ≤ 9) → (0 < c ∧ c ≤ 9) → 
  a ≠ b → b ≠ c → a ≠ c →
  (100*a + 10*b + c) + (100*a + 10*c + b) + (100*b + 10*a + c) + 
  (100*b + 10*c + a) + (100*c + 10*a + b) + (100*c + 10*b + a) ≠ 2018 :=
sorry

end sum_of_three_digit_permutations_not_2018_l2399_239980


namespace episode_length_l2399_239906

def total_days : ℕ := 5
def episodes : ℕ := 20
def daily_hours : ℕ := 2
def minutes_per_hour : ℕ := 60

theorem episode_length :
  (total_days * daily_hours * minutes_per_hour) / episodes = 30 := by
  sorry

end episode_length_l2399_239906


namespace largest_domain_of_g_l2399_239902

def g_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → g x + g (1 / x) = x^2

theorem largest_domain_of_g :
  ∃! (S : Set ℝ), S.Nonempty ∧
    (∀ T : Set ℝ, (∃ g : ℝ → ℝ, (∀ x ∈ T, x ≠ 0 ∧ g_condition g) → T ⊆ S)) ∧
    S = {-1, 1} :=
  sorry

end largest_domain_of_g_l2399_239902


namespace annual_income_proof_l2399_239946

/-- Calculates the yearly simple interest income given principal and rate -/
def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem annual_income_proof (total_amount : ℝ) (part1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : total_amount = 2500)
  (h2 : part1 = 1000)
  (h3 : rate1 = 0.05)
  (h4 : rate2 = 0.06) :
  simple_interest part1 rate1 + simple_interest (total_amount - part1) rate2 = 140 := by
  sorry

end annual_income_proof_l2399_239946


namespace not_all_two_equal_sides_congruent_l2399_239904

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  is_right : leg1^2 + leg2^2 = hypotenuse^2

-- Define congruence for right triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Statement to be proven false
theorem not_all_two_equal_sides_congruent :
  ¬ (∀ t1 t2 : RightTriangle,
    (t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) ∨
    (t1.leg1 = t2.leg1 ∧ t1.hypotenuse = t2.hypotenuse) ∨
    (t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse)
    → congruent t1 t2) :=
  sorry

end not_all_two_equal_sides_congruent_l2399_239904


namespace tangent_line_condition_l2399_239927

/-- The function f(x) = 2x - a ln x has a tangent line y = x + 1 at the point (1, f(1)) if and only if a = 1 -/
theorem tangent_line_condition (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 2*x - a * Real.log x) ∧ 
   (∃ g : ℝ → ℝ, (∀ x, g x = x + 1) ∧ 
    (∀ h : ℝ → ℝ, HasDerivAt f (g 1 - f 1) 1 → h = g))) ↔ 
  a = 1 := by sorry

end tangent_line_condition_l2399_239927


namespace soccer_league_games_l2399_239941

/-- Proves that in a soccer league with 11 teams and 55 total games, each team plays others 2 times -/
theorem soccer_league_games (num_teams : ℕ) (total_games : ℕ) (games_per_pair : ℕ) : 
  num_teams = 11 → 
  total_games = 55 → 
  total_games = (num_teams * (num_teams - 1) * games_per_pair) / 2 → 
  games_per_pair = 2 := by
sorry

end soccer_league_games_l2399_239941


namespace victory_points_value_l2399_239983

/-- Represents the number of points awarded for different match outcomes -/
structure PointSystem where
  victory : ℕ
  draw : ℕ
  defeat : ℕ

/-- Represents the state of a team's performance in the tournament -/
structure TeamPerformance where
  totalMatches : ℕ
  playedMatches : ℕ
  currentPoints : ℕ
  pointsNeeded : ℕ
  minWinsNeeded : ℕ

/-- The theorem stating the point value for a victory -/
theorem victory_points_value (ps : PointSystem) (tp : TeamPerformance) : 
  ps.draw = 1 ∧ 
  ps.defeat = 0 ∧
  tp.totalMatches = 20 ∧
  tp.playedMatches = 5 ∧
  tp.currentPoints = 12 ∧
  tp.pointsNeeded = 40 ∧
  tp.minWinsNeeded = 7 →
  ps.victory = 4 := by
  sorry

end victory_points_value_l2399_239983


namespace union_of_sets_l2399_239947

theorem union_of_sets (M N : Set ℕ) : 
  M = {0, 2, 3} → N = {1, 3} → M ∪ N = {0, 1, 2, 3} := by
  sorry

end union_of_sets_l2399_239947


namespace sticker_distribution_l2399_239901

/-- The number of ways to partition n identical objects into k or fewer non-negative integer parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 30 ways to partition 10 identical objects into 5 or fewer parts -/
theorem sticker_distribution : partition_count 10 5 = 30 := by sorry

end sticker_distribution_l2399_239901


namespace move_right_four_units_l2399_239930

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally in a Cartesian coordinate system -/
def moveHorizontal (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The theorem stating that moving (-2, 3) 4 units right results in (2, 3) -/
theorem move_right_four_units :
  let initial : Point := { x := -2, y := 3 }
  let final : Point := moveHorizontal initial 4
  final.x = 2 ∧ final.y = 3 := by
  sorry

end move_right_four_units_l2399_239930


namespace train_journey_time_l2399_239988

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  (4/5 * usual_speed) * (usual_time + 3/4) = usual_speed * usual_time → 
  usual_time = 3 := by
  sorry

end train_journey_time_l2399_239988


namespace cube_stacking_height_l2399_239938

/-- The edge length of the large cube in meters -/
def large_cube_edge : ℝ := 1

/-- The edge length of the small cubes in millimeters -/
def small_cube_edge : ℝ := 1

/-- Conversion factor from meters to millimeters -/
def m_to_mm : ℝ := 1000

/-- Conversion factor from kilometers to millimeters -/
def km_to_mm : ℝ := 1000000

/-- The height of the column formed by stacking all small cubes in kilometers -/
def column_height : ℝ := 1000

theorem cube_stacking_height :
  (large_cube_edge * m_to_mm)^3 / small_cube_edge^3 * small_cube_edge / km_to_mm = column_height := by
  sorry

end cube_stacking_height_l2399_239938


namespace relationship_between_x_and_y_l2399_239962

theorem relationship_between_x_and_y (x y m : ℝ) 
  (hx : x = 3 - m) (hy : y = 2*m + 1) : 2*x + y = 7 := by
  sorry

end relationship_between_x_and_y_l2399_239962


namespace opposite_sign_sum_zero_l2399_239944

theorem opposite_sign_sum_zero (a b : ℝ) : 
  (|a - 2| + (b + 1)^2 = 0) → (a - b = 3) := by
  sorry

end opposite_sign_sum_zero_l2399_239944


namespace triangle_area_l2399_239909

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (f A = 2) ∧
  (a = Real.sqrt 7) ∧
  (Real.sin B = 2 * Real.sin C) ∧
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (1/2 * a * b * Real.sin C) = 7 * Real.sqrt 3 / 6 := by
  sorry

end triangle_area_l2399_239909


namespace leftover_grass_seed_coverage_l2399_239931

/-- Proves the leftover grass seed coverage for Drew's lawn -/
theorem leftover_grass_seed_coverage 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) 
  (seed_bags : ℕ) 
  (coverage_per_bag : ℕ) :
  lawn_length = 22 →
  lawn_width = 36 →
  seed_bags = 4 →
  coverage_per_bag = 250 →
  seed_bags * coverage_per_bag - lawn_length * lawn_width = 208 :=
by sorry

end leftover_grass_seed_coverage_l2399_239931


namespace f_13_equals_219_l2399_239916

def f (n : ℕ) : ℕ := n^2 + 3*n + 11

theorem f_13_equals_219 : f 13 = 219 := by sorry

end f_13_equals_219_l2399_239916


namespace fraction_evaluation_l2399_239984

theorem fraction_evaluation (a b c : ℝ) (ha : a = 4) (hb : b = -4) (hc : c = 3) :
  3 / (a + b + c) = 1 := by
  sorry

end fraction_evaluation_l2399_239984


namespace austin_hourly_rate_l2399_239953

def hours_per_week : ℕ := 6
def weeks_worked : ℕ := 6
def bicycle_cost : ℕ := 180

theorem austin_hourly_rate :
  ∃ (rate : ℚ), rate * (hours_per_week * weeks_worked : ℚ) = bicycle_cost ∧ rate = 5 := by
  sorry

end austin_hourly_rate_l2399_239953


namespace negation_of_exists_leq_negation_of_proposition_l2399_239954

theorem negation_of_exists_leq (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end negation_of_exists_leq_negation_of_proposition_l2399_239954


namespace puzzle_solution_l2399_239913

-- Define the grid type
def Grid := Matrix (Fin 6) (Fin 6) Nat

-- Define the constraint for black dots (ratio of 2)
def blackDotConstraint (a b : Nat) : Prop := a = 2 * b ∨ b = 2 * a

-- Define the constraint for white dots (difference of 1)
def whiteDotConstraint (a b : Nat) : Prop := a = b + 1 ∨ b = a + 1

-- Define the property of having no repeated numbers in a row or column
def noRepeats (g : Grid) : Prop :=
  ∀ i j : Fin 6, i ≠ j → 
    (∀ k : Fin 6, g i k ≠ g j k) ∧ 
    (∀ k : Fin 6, g k i ≠ g k j)

-- Define the property that all numbers are between 1 and 6
def validNumbers (g : Grid) : Prop :=
  ∀ i j : Fin 6, 1 ≤ g i j ∧ g i j ≤ 6

-- Define the specific constraints for this puzzle
def puzzleConstraints (g : Grid) : Prop :=
  blackDotConstraint (g 0 0) (g 0 1) ∧
  whiteDotConstraint (g 0 4) (g 0 5) ∧
  blackDotConstraint (g 1 2) (g 1 3) ∧
  whiteDotConstraint (g 2 1) (g 2 2) ∧
  blackDotConstraint (g 3 0) (g 3 1) ∧
  whiteDotConstraint (g 3 2) (g 3 3) ∧
  blackDotConstraint (g 4 4) (g 4 5) ∧
  whiteDotConstraint (g 5 3) (g 5 4)

-- Theorem statement
theorem puzzle_solution :
  ∀ g : Grid,
    noRepeats g →
    validNumbers g →
    puzzleConstraints g →
    g 3 0 = 2 ∧ g 3 1 = 1 ∧ g 3 2 = 4 ∧ g 3 3 = 3 ∧ g 3 4 = 6 :=
sorry

end puzzle_solution_l2399_239913


namespace quadratic_equation_solution_l2399_239978

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -3/2 ∧ 2*x₁^2 - 4*x₁ = 6 - 3*x₁) ∧
  (x₂ = 2 ∧ 2*x₂^2 - 4*x₂ = 6 - 3*x₂) := by
  sorry

end quadratic_equation_solution_l2399_239978


namespace smores_group_size_l2399_239921

/-- Given the conditions for S'mores supplies, prove the number of people in the group. -/
theorem smores_group_size :
  ∀ (smores_per_person : ℕ) 
    (cost_per_set : ℕ) 
    (smores_per_set : ℕ) 
    (total_cost : ℕ),
  smores_per_person = 3 →
  cost_per_set = 3 →
  smores_per_set = 4 →
  total_cost = 18 →
  (total_cost / cost_per_set) * smores_per_set / smores_per_person = 8 :=
by sorry

end smores_group_size_l2399_239921


namespace lcm_hcf_problem_l2399_239959

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 47 → 
  b = 517 → 
  a = 210 := by sorry

end lcm_hcf_problem_l2399_239959


namespace certain_inning_is_19th_l2399_239971

/-- Represents the statistics of a cricketer before and after a certain inning -/
structure CricketerStats where
  prevInnings : ℕ
  prevAverage : ℚ
  runsScored : ℕ
  newAverage : ℚ

/-- Theorem stating that given the conditions, the certain inning was the 19th inning -/
theorem certain_inning_is_19th (stats : CricketerStats)
  (h1 : stats.runsScored = 97)
  (h2 : stats.newAverage = stats.prevAverage + 4)
  (h3 : stats.newAverage = 25) :
  stats.prevInnings + 1 = 19 := by
  sorry

end certain_inning_is_19th_l2399_239971


namespace max_value_rational_function_l2399_239977

theorem max_value_rational_function (x : ℝ) :
  x^4 / (x^8 + 2*x^6 + 4*x^4 + 8*x^2 + 16) ≤ 1/20 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 + 4*y^4 + 8*y^2 + 16) = 1/20 :=
by sorry

end max_value_rational_function_l2399_239977


namespace square_of_product_l2399_239950

theorem square_of_product (x : ℝ) : (3 * x)^2 = 9 * x^2 := by
  sorry

end square_of_product_l2399_239950


namespace decagon_diagonals_from_vertex_l2399_239933

/-- The number of diagonals from one vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals_from_vertex :
  diagonals_from_vertex decagon_sides = 7 := by
  sorry

end decagon_diagonals_from_vertex_l2399_239933


namespace parallelogram_area_32_15_l2399_239969

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 15 cm is 480 square centimeters -/
theorem parallelogram_area_32_15 :
  parallelogram_area 32 15 = 480 := by
  sorry

end parallelogram_area_32_15_l2399_239969


namespace prob_sum_32_four_eight_sided_dice_prob_sum_32_four_eight_sided_dice_eq_frac_l2399_239979

/-- The probability of rolling a sum of 32 with four fair eight-sided dice -/
theorem prob_sum_32_four_eight_sided_dice : ℝ :=
  let num_faces : ℕ := 8
  let num_dice : ℕ := 4
  let target_sum : ℕ := 32
  let prob_max_face : ℝ := 1 / num_faces
  (prob_max_face ^ num_dice : ℝ)

#check prob_sum_32_four_eight_sided_dice

theorem prob_sum_32_four_eight_sided_dice_eq_frac :
  prob_sum_32_four_eight_sided_dice = 1 / 4096 := by sorry

end prob_sum_32_four_eight_sided_dice_prob_sum_32_four_eight_sided_dice_eq_frac_l2399_239979


namespace orthocenter_locus_l2399_239997

noncomputable section

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is inside a circle --/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Check if a triangle is inscribed in a circle --/
def is_inscribed (t : Triangle) (c : Circle) : Prop := sorry

/-- The theorem stating the locus of orthocenters --/
theorem orthocenter_locus (c : Circle) :
  ∀ t : Triangle, is_inscribed t c →
    is_inside (orthocenter t) { center := c.center, radius := 3 * c.radius } :=
sorry

end orthocenter_locus_l2399_239997


namespace base_7_representation_of_864_base_7_correctness_l2399_239951

/-- Converts a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_of_864 :
  toBase7 864 = [2, 3, 4, 3] :=
sorry

theorem base_7_correctness :
  fromBase7 [2, 3, 4, 3] = 864 :=
sorry

end base_7_representation_of_864_base_7_correctness_l2399_239951


namespace functional_equation_solution_l2399_239918

/-- A function satisfying the given functional equation is constant and equal to 2 -/
theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x > 0, f x > 0) → 
  (∀ x y, x > 0 → y > 0 → f x * f y = 2 * f (x + y * f x)) → 
  (∀ x > 0, f x = 2) := by
sorry

end functional_equation_solution_l2399_239918


namespace min_value_problem_l2399_239998

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ -2048113 := by
  sorry

end min_value_problem_l2399_239998


namespace min_value_expression_l2399_239942

theorem min_value_expression (x y : ℝ) 
  (h1 : |x| < 1) 
  (h2 : |y| < 2) 
  (h3 : x * y = 1) : 
  (1 / (1 - x^2)) + (4 / (4 - y^2)) ≥ 4 ∧ 
  ∃ (x₀ y₀ : ℝ), |x₀| < 1 ∧ |y₀| < 2 ∧ x₀ * y₀ = 1 ∧ 
    (1 / (1 - x₀^2)) + (4 / (4 - y₀^2)) = 4 :=
by sorry

end min_value_expression_l2399_239942


namespace plane_equation_proof_l2399_239982

/-- A plane in 3D space represented by the equation Ax + By + Cz + D = 0 --/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def parallelPlanes (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = 2 ∧ given_plane.B = -1 ∧ given_plane.C = 3 ∧ given_plane.D = 5 →
  point.x = 2 ∧ point.y = 3 ∧ point.z = -4 →
  ∃ (result_plane : Plane),
    parallelPlanes result_plane given_plane ∧
    pointOnPlane point result_plane ∧
    result_plane.A = 2 ∧ result_plane.B = -1 ∧ result_plane.C = 3 ∧ result_plane.D = 11 :=
sorry

end plane_equation_proof_l2399_239982


namespace racket_sales_revenue_l2399_239963

theorem racket_sales_revenue 
  (average_price : ℝ) 
  (pairs_sold : ℕ) 
  (h1 : average_price = 9.8) 
  (h2 : pairs_sold = 75) :
  average_price * (pairs_sold : ℝ) = 735 := by
  sorry

end racket_sales_revenue_l2399_239963


namespace line_inclination_angle_l2399_239965

theorem line_inclination_angle (x1 y1 x2 y2 : ℝ) :
  x1 = 1 →
  y1 = 1 →
  x2 = 2 →
  y2 = 1 + Real.sqrt 3 →
  ∃ θ : ℝ, θ * (π / 180) = π / 3 ∧ Real.tan θ = (y2 - y1) / (x2 - x1) := by
  sorry

end line_inclination_angle_l2399_239965


namespace arithmetic_sequence_sum_l2399_239974

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- sum formula
  3 * a 5 - a 1 = 10 →  -- given condition
  S 13 = 117 := by
sorry

end arithmetic_sequence_sum_l2399_239974


namespace imaginary_part_of_z_l2399_239949

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((1 - 2*i) / (2 + i)) = -1 := by
  sorry

end imaginary_part_of_z_l2399_239949


namespace line_circle_intersection_x_intercept_l2399_239923

/-- The x-intercept of a line that intersects a circle --/
theorem line_circle_intersection_x_intercept
  (m : ℝ)  -- Slope of the line
  (h1 : ∀ x y : ℝ, m * x + y + 3 * m - Real.sqrt 3 = 0 → x^2 + y^2 = 12 → 
         ∃ A B : ℝ × ℝ, A ≠ B ∧ 
         m * A.1 + A.2 + 3 * m - Real.sqrt 3 = 0 ∧
         A.1^2 + A.2^2 = 12 ∧
         m * B.1 + B.2 + 3 * m - Real.sqrt 3 = 0 ∧
         B.1^2 + B.2^2 = 12)
  (h2 : ∃ A B : ℝ × ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) :
  ∃ x : ℝ, x = -6 ∧ m * x + 3 * m - Real.sqrt 3 = 0 :=
sorry

end line_circle_intersection_x_intercept_l2399_239923


namespace A_necessary_not_sufficient_l2399_239973

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define propositions A and B
def proposition_A (x : ℝ) : Prop := log10 (x^2) = 0
def proposition_B (x : ℝ) : Prop := x = 1

-- Theorem stating A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∀ x : ℝ, proposition_B x → proposition_A x) ∧
  (∃ x : ℝ, proposition_A x ∧ ¬proposition_B x) :=
sorry

end A_necessary_not_sufficient_l2399_239973


namespace frisbee_price_problem_l2399_239912

/-- Proves that the certain price of frisbees is $4 given the problem conditions -/
theorem frisbee_price_problem (total_frisbees : ℕ) (price_some : ℝ) (price_rest : ℝ) 
  (total_receipts : ℝ) (min_at_price_rest : ℕ) :
  total_frisbees = 60 →
  price_some = 3 →
  total_receipts = 200 →
  min_at_price_rest = 20 →
  price_rest = 4 := by
  sorry

end frisbee_price_problem_l2399_239912


namespace cost_of_500_pencils_is_25_dollars_l2399_239991

/-- The cost of 500 pencils in dollars -/
def cost_of_500_pencils : ℚ :=
  let cost_per_pencil : ℚ := 5 / 100  -- 5 cents in dollars
  let number_of_pencils : ℕ := 500
  cost_per_pencil * number_of_pencils

theorem cost_of_500_pencils_is_25_dollars :
  cost_of_500_pencils = 25 := by sorry

end cost_of_500_pencils_is_25_dollars_l2399_239991


namespace absolute_value_equation_solution_l2399_239939

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |1/2 * x| = 2 ↔ x = 4 ∨ x = -4 := by
sorry

end absolute_value_equation_solution_l2399_239939


namespace children_neither_happy_nor_sad_l2399_239936

theorem children_neither_happy_nor_sad 
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 19)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_boys = 7)
  : total_children - happy_children - sad_children = 20 := by
  sorry

end children_neither_happy_nor_sad_l2399_239936


namespace carly_running_schedule_l2399_239964

def running_schedule (week1 : ℚ) (week2_multiplier : ℚ) (week2_extra : ℚ) (week3_multiplier : ℚ) (week4_reduction : ℚ) : ℚ → ℚ
  | 1 => week1
  | 2 => week1 * week2_multiplier + week2_extra
  | 3 => (week1 * week2_multiplier + week2_extra) * week3_multiplier
  | 4 => (week1 * week2_multiplier + week2_extra) * week3_multiplier - week4_reduction
  | _ => 0

theorem carly_running_schedule :
  running_schedule 2 2 3 (9/7) 5 4 = 4 := by sorry

end carly_running_schedule_l2399_239964


namespace sin_product_equals_one_sixteenth_l2399_239985

theorem sin_product_equals_one_sixteenth :
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 16 :=
by sorry

end sin_product_equals_one_sixteenth_l2399_239985


namespace eight_couples_handshakes_l2399_239926

/-- The number of handshakes in a gathering of couples where each person
    shakes hands with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  (2 * n) * (2 * n - 2) / 2

/-- Theorem: In a gathering of 8 couples, if each person shakes hands once
    with everyone except their spouse, the total number of handshakes is 112 -/
theorem eight_couples_handshakes :
  handshakes 8 = 112 := by
  sorry

#eval handshakes 8  -- Should output 112

end eight_couples_handshakes_l2399_239926


namespace p_sufficient_not_necessary_for_q_l2399_239911

theorem p_sufficient_not_necessary_for_q :
  (∃ x, 0 < x ∧ x < 5 ∧ ¬(-1 < x ∧ x < 5)) = False ∧
  (∃ x, -1 < x ∧ x < 5 ∧ ¬(0 < x ∧ x < 5)) = True := by
  sorry

end p_sufficient_not_necessary_for_q_l2399_239911


namespace fraction_equality_l2399_239905

theorem fraction_equality (a b : ℝ) (h : 2/a - 1/b = 1/(a + 2*b)) :
  4/a^2 - 1/b^2 = 1/(a*b) := by sorry

end fraction_equality_l2399_239905


namespace trailing_zeros_of_square_l2399_239975

theorem trailing_zeros_of_square : ∃ n : ℕ, (10^11 - 2)^2 = n * 10^10 ∧ n % 10 ≠ 0 := by
  sorry

end trailing_zeros_of_square_l2399_239975


namespace more_philosophers_than_mathematicians_l2399_239929

theorem more_philosophers_than_mathematicians
  (m p : ℕ+)
  (h : (m : ℚ) / 7 = (p : ℚ) / 9) :
  p > m :=
sorry

end more_philosophers_than_mathematicians_l2399_239929


namespace purely_imaginary_implies_m_eq_two_third_quadrant_implies_m_range_l2399_239987

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m^2 + 2*m - 3)

-- Define the condition for z - m + 2 being purely imaginary
def is_purely_imaginary (m : ℝ) : Prop :=
  (z m - m + 2).re = 0 ∧ (z m - m + 2).im ≠ 0

-- Define the condition for point A being in the third quadrant
def in_third_quadrant (m : ℝ) : Prop :=
  (z m).re < 0 ∧ (z m).im < 0

-- Theorem 1: If z - m + 2 is purely imaginary, then m = 2
theorem purely_imaginary_implies_m_eq_two (m : ℝ) :
  is_purely_imaginary m → m = 2 :=
sorry

-- Theorem 2: If point A is in the third quadrant, then -3 < m < 1
theorem third_quadrant_implies_m_range (m : ℝ) :
  in_third_quadrant m → -3 < m ∧ m < 1 :=
sorry

end purely_imaginary_implies_m_eq_two_third_quadrant_implies_m_range_l2399_239987


namespace cube_preserves_order_for_negative_numbers_l2399_239993

theorem cube_preserves_order_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 := by
  sorry

end cube_preserves_order_for_negative_numbers_l2399_239993


namespace xy_squared_minus_y_squared_x_equals_zero_l2399_239976

theorem xy_squared_minus_y_squared_x_equals_zero (x y : ℝ) : x * y^2 - y^2 * x = 0 := by
  sorry

end xy_squared_minus_y_squared_x_equals_zero_l2399_239976


namespace stratified_sample_small_supermarkets_l2399_239999

/-- Calculates the number of small supermarkets in a stratified sample -/
def smallSupermarketsInSample (totalSupermarkets : ℕ) (smallSupermarkets : ℕ) (sampleSize : ℕ) : ℕ :=
  (smallSupermarkets * sampleSize) / totalSupermarkets

theorem stratified_sample_small_supermarkets :
  smallSupermarketsInSample 3000 2100 100 = 70 := by
  sorry

end stratified_sample_small_supermarkets_l2399_239999
