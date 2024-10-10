import Mathlib

namespace ellipse_eccentricity_l3123_312311

/-- Given an ellipse and a circle satisfying certain conditions, prove that the eccentricity of the ellipse is 1/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ 
   ((x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 + y^2 = a^2) ∨ 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 - c^2 = a^2 - c^2 ∧ c^2 = a^2 - b^2))) →
  (a^2 - b^2) / a = 1/3 :=
sorry

end ellipse_eccentricity_l3123_312311


namespace vegetarian_count_l3123_312340

theorem vegetarian_count (non_veg_only : ℕ) (both : ℕ) (total_veg : ℕ) 
  (h1 : non_veg_only = 9)
  (h2 : both = 12)
  (h3 : total_veg = 28) :
  total_veg - both = 16 := by
  sorry

end vegetarian_count_l3123_312340


namespace original_number_property_l3123_312320

theorem original_number_property (k : ℕ) : ∃ (N : ℕ), N = 23 * k + 22 ∧ (N + 1) % 23 = 0 := by
  sorry

end original_number_property_l3123_312320


namespace converse_proposition_l3123_312379

theorem converse_proposition : 
  (∀ x : ℝ, x = 3 → x^2 - 2*x - 3 = 0) ↔ 
  (∀ x : ℝ, x^2 - 2*x - 3 = 0 → x = 3) :=
by sorry

end converse_proposition_l3123_312379


namespace absolute_value_inequality_l3123_312339

theorem absolute_value_inequality (x : ℝ) :
  (1 < |x - 1| ∧ |x - 1| < 4) ↔ ((-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5)) := by
  sorry

end absolute_value_inequality_l3123_312339


namespace digit_150_is_5_l3123_312328

-- Define the fraction
def fraction : ℚ := 5 / 37

-- Define the length of the repeating cycle
def cycle_length : ℕ := 3

-- Define the position we're interested in
def target_position : ℕ := 150

-- Define the function to get the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_150_is_5 : nth_digit target_position = 5 := by sorry

end digit_150_is_5_l3123_312328


namespace equation_solution_l3123_312369

theorem equation_solution : 
  ∀ x : ℝ, (2010 + x)^2 = 4*x^2 ↔ x = 2010 ∨ x = -670 := by
sorry

end equation_solution_l3123_312369


namespace trailing_zeros_factorial_100_l3123_312352

-- Define a function to count trailing zeros in factorial
def trailingZerosInFactorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Theorem statement
theorem trailing_zeros_factorial_100 : trailingZerosInFactorial 100 = 24 := by
  sorry


end trailing_zeros_factorial_100_l3123_312352


namespace not_less_than_negative_double_l3123_312357

theorem not_less_than_negative_double {x y : ℝ} (h : x < y) : ¬(-2 * x < -2 * y) := by
  sorry

end not_less_than_negative_double_l3123_312357


namespace faith_works_five_days_l3123_312321

/-- Faith's work schedule and earnings --/
def faith_work_schedule (hourly_rate : ℚ) (regular_hours : ℕ) (overtime_hours : ℕ) (weekly_earnings : ℚ) : Prop :=
  ∃ (days_worked : ℕ),
    (hourly_rate * regular_hours + hourly_rate * 1.5 * overtime_hours) * days_worked = weekly_earnings ∧
    days_worked ≤ 7

theorem faith_works_five_days :
  faith_work_schedule 13.5 8 2 675 →
  ∃ (days_worked : ℕ), days_worked = 5 := by
  sorry

end faith_works_five_days_l3123_312321


namespace fraction_puzzle_l3123_312382

theorem fraction_puzzle (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : 
  min x y = 1/4 := by
sorry

end fraction_puzzle_l3123_312382


namespace complex_roots_theorem_l3123_312333

theorem complex_roots_theorem (p q r : ℂ) 
  (sum_eq : p + q + r = -1)
  (sum_prod_eq : p * q + p * r + q * r = -1)
  (prod_eq : p * q * r = -1) :
  (({p, q, r} : Set ℂ) = {-1, Complex.I, -Complex.I}) := by
  sorry

end complex_roots_theorem_l3123_312333


namespace gcd_g_x_eq_one_l3123_312304

def g (x : ℤ) : ℤ := (3*x+4)*(9*x+5)*(17*x+11)*(x+17)

theorem gcd_g_x_eq_one (x : ℤ) (h : ∃ k : ℤ, x = 7263 * k) :
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 1 := by
  sorry

end gcd_g_x_eq_one_l3123_312304


namespace circle_passes_through_P_with_center_C_l3123_312301

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 29

-- Define the center point
def center : ℝ × ℝ := (3, 0)

-- Define the point P
def point_P : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem circle_passes_through_P_with_center_C :
  circle_equation point_P.1 point_P.2 ∧
  ∀ (x y : ℝ), circle_equation x y → 
    (x - center.1)^2 + (y - center.2)^2 = 
    (point_P.1 - center.1)^2 + (point_P.2 - center.2)^2 := by
  sorry

end circle_passes_through_P_with_center_C_l3123_312301


namespace greatest_x_lcm_l3123_312307

theorem greatest_x_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 10 14) = 70) → x ≤ 70 ∧ ∃ y : ℕ, y = 70 ∧ Nat.lcm y (Nat.lcm 10 14) = 70 :=
by sorry

end greatest_x_lcm_l3123_312307


namespace cricket_count_l3123_312302

theorem cricket_count (initial : Real) (additional : Real) :
  initial = 7.0 → additional = 11.0 → initial + additional = 18.0 := by
  sorry

end cricket_count_l3123_312302


namespace point_coordinates_l3123_312334

theorem point_coordinates (m n : ℝ) : (m + 3)^2 + Real.sqrt (4 - n) = 0 → m = -3 ∧ n = 4 := by
  sorry

end point_coordinates_l3123_312334


namespace scientific_notation_12000_l3123_312318

theorem scientific_notation_12000 : 
  12000 = 1.2 * (10 : ℝ)^4 := by sorry

end scientific_notation_12000_l3123_312318


namespace focus_line_dot_product_fixed_point_existence_l3123_312384

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line that intersects the parabola at two distinct points
def intersecting_line (t b : ℝ) (x y : ℝ) : Prop := x = t*y + b

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

-- Part I: Theorem for line passing through focus
theorem focus_line_dot_product (t : ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  intersecting_line t 1 x1 y1 ∧ intersecting_line t 1 x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) →
  dot_product x1 y1 x2 y2 = -3 :=
sorry

-- Part II: Theorem for fixed point
theorem fixed_point_existence (t b : ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  intersecting_line t b x1 y1 ∧ intersecting_line t b x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧
  dot_product x1 y1 x2 y2 = -4 →
  b = 2 :=
sorry

end focus_line_dot_product_fixed_point_existence_l3123_312384


namespace coin_game_probability_l3123_312343

/-- Represents a player in the coin game -/
inductive Player := | Abby | Bernardo | Carl | Debra

/-- Represents a ball color in the game -/
inductive BallColor := | Green | Red | Blue

/-- The number of rounds in the game -/
def numRounds : Nat := 5

/-- The number of coins each player starts with -/
def initialCoins : Nat := 5

/-- The number of balls of each color in the urn -/
def ballCounts : Fin 3 → Nat
  | 0 => 2  -- Green
  | 1 => 2  -- Red
  | 2 => 1  -- Blue

/-- Represents the state of the game after each round -/
structure GameState where
  coins : Player → Nat
  round : Nat

/-- Represents a single round of the game -/
def gameRound (state : GameState) : GameState := sorry

/-- The probability of a specific outcome in a single round -/
def roundProbability (outcome : Player → BallColor) : Rat := sorry

/-- The probability of returning to the initial state after all rounds -/
def finalProbability : Rat := sorry

/-- The main theorem stating the probability of each player having 5 coins at the end -/
theorem coin_game_probability : finalProbability = 64 / 15625 := sorry

end coin_game_probability_l3123_312343


namespace bobby_sarah_fish_ratio_l3123_312354

/-- The number of fish in each person's aquarium and their relationships -/
structure FishCounts where
  billy : ℕ
  tony : ℕ
  sarah : ℕ
  bobby : ℕ
  billy_count : billy = 10
  tony_count : tony = 3 * billy
  sarah_count : sarah = tony + 5
  total_count : billy + tony + sarah + bobby = 145

/-- The ratio of fish in Bobby's aquarium to Sarah's aquarium -/
def fish_ratio (fc : FishCounts) : ℚ :=
  fc.bobby / fc.sarah

/-- Theorem stating that the ratio of fish in Bobby's aquarium to Sarah's aquarium is 2:1 -/
theorem bobby_sarah_fish_ratio (fc : FishCounts) : fish_ratio fc = 2 / 1 := by
  sorry

end bobby_sarah_fish_ratio_l3123_312354


namespace f_properties_l3123_312393

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (f (25 * Real.pi / 6) = 0) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ x_max : ℝ, x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f x_max = (2 - Real.sqrt 3) / 2 ∧
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x_max) ∧
  (∃ x_min : ℝ, x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    f x_min = -Real.sqrt 3 ∧
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x_min ≤ f x) :=
by sorry

#check f_properties

end f_properties_l3123_312393


namespace cube_sum_inequality_l3123_312378

theorem cube_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b*c)) + (b^3 / (c*a)) + (c^3 / (a*b)) ≥ a + b + c := by
  sorry

end cube_sum_inequality_l3123_312378


namespace max_value_fraction_l3123_312348

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b + b * c) / (a^2 + b^2 + c^2) = Real.sqrt 2 / 2 :=
sorry

end max_value_fraction_l3123_312348


namespace train_vs_airplanes_capacity_difference_l3123_312331

/-- The number of passengers a single train car can carry -/
def train_car_capacity : ℕ := 60

/-- The number of passengers a 747 airplane can carry -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The number of airplanes being compared -/
def num_airplanes : ℕ := 2

/-- Theorem stating the difference in passenger capacity between the train and the airplanes -/
theorem train_vs_airplanes_capacity_difference :
  train_cars * train_car_capacity - num_airplanes * airplane_capacity = 228 := by
  sorry

end train_vs_airplanes_capacity_difference_l3123_312331


namespace certain_number_is_seven_l3123_312395

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem certain_number_is_seven (n : ℕ) (h : factorial 9 / factorial n = 72) : n = 7 := by
  sorry

end certain_number_is_seven_l3123_312395


namespace binary_101110110101_mod_8_l3123_312317

/-- Given a binary number represented as a list of bits (least significant bit first),
    calculate its value modulo 8. -/
def binary_mod_8 (bits : List Bool) : Nat :=
  match bits.take 3 with
  | [b0, b1, b2] => (if b0 then 1 else 0) + (if b1 then 2 else 0) + (if b2 then 4 else 0)
  | _ => 0

theorem binary_101110110101_mod_8 :
  binary_mod_8 [true, false, true, false, true, true, false, true, true, false, true, true] = 5 := by
  sorry

end binary_101110110101_mod_8_l3123_312317


namespace valid_attachments_count_l3123_312351

/-- Represents a square in our figure -/
structure Square

/-- Represents the cross-shaped figure -/
structure CrossFigure where
  center : Square
  extensions : Fin 4 → Square

/-- Represents a position where an extra square can be attached -/
inductive AttachmentPosition
  | TopOfExtension (i : Fin 4)
  | Other

/-- Represents the resulting figure after attaching an extra square -/
structure ResultingFigure where
  base : CrossFigure
  extraSquare : Square
  position : AttachmentPosition

/-- Predicate to check if a resulting figure can be folded into a topless square pyramid -/
def canFoldIntoPyramid (fig : ResultingFigure) : Prop :=
  match fig.position with
  | AttachmentPosition.TopOfExtension _ => True
  | AttachmentPosition.Other => False

/-- The main theorem to prove -/
theorem valid_attachments_count :
  ∃ (validPositions : Finset AttachmentPosition),
    (∀ pos, pos ∈ validPositions ↔ ∃ fig : ResultingFigure, fig.position = pos ∧ canFoldIntoPyramid fig) ∧
    Finset.card validPositions = 4 := by
  sorry

end valid_attachments_count_l3123_312351


namespace average_weight_problem_l3123_312330

theorem average_weight_problem (total_boys : Nat) (group_a_boys : Nat) (group_b_boys : Nat)
  (group_b_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = 34 →
  group_a_boys = 26 →
  group_b_boys = 8 →
  group_b_avg_weight = 45.15 →
  total_avg_weight = 49.05 →
  let group_a_avg_weight := (total_boys * total_avg_weight - group_b_boys * group_b_avg_weight) / group_a_boys
  group_a_avg_weight = 50.25 := by
  sorry

end average_weight_problem_l3123_312330


namespace inscribed_quadrilateral_fourth_side_l3123_312322

/-- A quadrilateral inscribed in a circle with given side lengths --/
structure InscribedQuadrilateral where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem stating the relationship between the sides of the inscribed quadrilateral --/
theorem inscribed_quadrilateral_fourth_side 
  (q : InscribedQuadrilateral) 
  (h1 : q.radius = 100 * Real.sqrt 3)
  (h2 : q.side1 = 100)
  (h3 : q.side2 = 150)
  (h4 : q.side3 = 200) :
  q.side4^2 = 35800 := by
  sorry

end inscribed_quadrilateral_fourth_side_l3123_312322


namespace rug_dimension_l3123_312398

theorem rug_dimension (x : ℝ) : 
  x > 0 ∧ 
  x ≤ 8 ∧
  7 ≤ 8 ∧
  x * 7 = 64 * (1 - 0.78125) →
  x = 2 := by
sorry

end rug_dimension_l3123_312398


namespace min_days_to_solve_100_problems_l3123_312341

/-- The number of problems solved on day n -/
def problems_solved (n : ℕ) : ℕ := 3^(n-1)

/-- The total number of problems solved up to day n -/
def total_problems (n : ℕ) : ℕ := (3^n - 1) / 2

theorem min_days_to_solve_100_problems :
  ∀ n : ℕ, n > 0 → (total_problems n ≥ 100 ↔ n ≥ 5) :=
sorry

end min_days_to_solve_100_problems_l3123_312341


namespace triangle_inequality_range_l3123_312312

/-- A right-angled triangle with sides a, b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  pythagorean : a^2 + b^2 = c^2

/-- The theorem stating the range of t for which the given inequality holds -/
theorem triangle_inequality_range (tri : RightTriangle) :
  (∀ t : ℝ, 1 / tri.a^2 + 4 / tri.b^2 + t / tri.c^2 ≥ 0) ↔ 
  (∀ t : ℝ, t ≥ -9 ∧ t ∈ Set.Ici (-9)) :=
sorry

end triangle_inequality_range_l3123_312312


namespace value_of_5_minus_c_l3123_312336

theorem value_of_5_minus_c (c d : ℤ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 7 + d = 10 + c) : 
  5 - c = 6 := by
  sorry

end value_of_5_minus_c_l3123_312336


namespace isosceles_triangle_base_length_l3123_312332

/-- Given an equilateral triangle and an isosceles triangle sharing a side,
    prove that the base of the isosceles triangle is 25 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral : equilateral_perimeter = 60)
  (h_isosceles : isosceles_perimeter = 65)
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) :
  isosceles_base = 25 :=
by
  sorry

#check isosceles_triangle_base_length

end isosceles_triangle_base_length_l3123_312332


namespace f_lower_bound_l3123_312394

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m / x + log x

theorem f_lower_bound (m : ℝ) (x : ℝ) (hm : m > 0) (hx : x > 0) :
  m * f m x ≥ 2 * m - 1 := by
  sorry

end f_lower_bound_l3123_312394


namespace right_triangle_check_other_sets_not_right_triangle_l3123_312324

theorem right_triangle_check (a b c : ℝ) : 
  (a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2 :=
by sorry

theorem other_sets_not_right_triangle :
  ¬(∃ a b c : ℝ, 
    ((a = Real.sqrt 3 ∧ b = Real.sqrt 4 ∧ c = Real.sqrt 5) ∨
     (a = 4 ∧ b = 9 ∧ c = Real.sqrt 13) ∨
     (a = 0.8 ∧ b = 0.15 ∧ c = 0.17)) ∧
    a^2 + b^2 = c^2) :=
by sorry

end right_triangle_check_other_sets_not_right_triangle_l3123_312324


namespace perimeter_decrease_percentage_l3123_312377

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Perimeter decrease percentage for different length and width reductions --/
theorem perimeter_decrease_percentage
  (r : Rectangle)
  (h1 : perimeter { length := 0.9 * r.length, width := 0.8 * r.width } = 0.88 * perimeter r) :
  perimeter { length := 0.8 * r.length, width := 0.9 * r.width } = 0.82 * perimeter r := by
  sorry


end perimeter_decrease_percentage_l3123_312377


namespace special_ellipse_equation_l3123_312367

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The center of the ellipse is at the origin -/
  center_at_origin : Unit
  /-- The endpoints of the minor axis are at (0, ±1) -/
  minor_axis_endpoints : Unit
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The product of the eccentricity of this ellipse and that of the hyperbola y^2 - x^2 = 1 is 1 -/
  eccentricity_product : e * Real.sqrt 2 = 1

/-- The equation of the special ellipse -/
def ellipse_equation (E : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Theorem stating that the given ellipse has the equation x^2/2 + y^2 = 1 -/
theorem special_ellipse_equation (E : SpecialEllipse) :
  ∀ x y : ℝ, ellipse_equation E x y :=
sorry

end special_ellipse_equation_l3123_312367


namespace candy_ratio_is_five_thirds_l3123_312374

/-- Represents the number of M&M candies Penelope has -/
def mm_candies : ℕ := 25

/-- Represents the number of Starbursts candies Penelope has -/
def starbursts_candies : ℕ := 15

/-- Represents the ratio of M&M candies to Starbursts candies -/
def candy_ratio : Rat := mm_candies / starbursts_candies

theorem candy_ratio_is_five_thirds : candy_ratio = 5 / 3 := by
  sorry

end candy_ratio_is_five_thirds_l3123_312374


namespace lines_parallel_in_plane_l3123_312344

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_in_plane 
  (m n : Line) (α β : Plane) :
  m ≠ n →  -- m and n are distinct
  α ≠ β →  -- α and β are distinct
  contained_in m α →
  parallel n α →
  coplanar m n β →
  parallel_lines m n :=
sorry

end lines_parallel_in_plane_l3123_312344


namespace tournament_games_count_l3123_312399

/-- Calculates the number of games in a round-robin tournament for a given number of teams -/
def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of games in knockout rounds for a given number of teams -/
def knockoutGames (n : ℕ) : ℕ := n - 1

theorem tournament_games_count :
  let totalTeams : ℕ := 32
  let groupCount : ℕ := 8
  let teamsPerGroup : ℕ := 4
  let advancingTeams : ℕ := 2
  
  totalTeams = groupCount * teamsPerGroup →
  
  (groupCount * roundRobinGames teamsPerGroup) +
  (knockoutGames (groupCount * advancingTeams)) = 63 := by
  sorry

end tournament_games_count_l3123_312399


namespace sin_1200_degrees_l3123_312388

theorem sin_1200_degrees : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_1200_degrees_l3123_312388


namespace sugar_water_and_triangle_inequalities_l3123_312389

theorem sugar_water_and_triangle_inequalities :
  (∀ x y m : ℝ, x > y ∧ y > 0 ∧ m > 0 → y / x < (y + m) / (x + m)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    a / (b + c) + b / (a + c) + c / (a + b) < 2) :=
by sorry

end sugar_water_and_triangle_inequalities_l3123_312389


namespace imaginary_part_of_i_times_one_plus_i_l3123_312390

theorem imaginary_part_of_i_times_one_plus_i (i : ℂ) : i * i = -1 → Complex.im (i * (1 + i)) = 1 := by
  sorry

end imaginary_part_of_i_times_one_plus_i_l3123_312390


namespace cow_feeding_problem_l3123_312373

theorem cow_feeding_problem (daily_feed : ℕ) (total_feed : ℕ) 
  (h1 : daily_feed = 28) (h2 : total_feed = 890) :
  ∃ (days : ℕ) (leftover : ℕ), 
    days * daily_feed + leftover = total_feed ∧ 
    days = 31 ∧ 
    leftover = 22 := by
  sorry

end cow_feeding_problem_l3123_312373


namespace f_neg_l3123_312380

-- Define an odd function f on the real numbers
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_pos : ∀ x > 0, f x = x * (1 - x)

-- Theorem to prove
theorem f_neg : ∀ x < 0, f x = x * (1 + x) := by sorry

end f_neg_l3123_312380


namespace first_divisor_problem_l3123_312306

theorem first_divisor_problem (x : ℚ) : 
  ((377 / x) / 29) * (1/4) / 2 = 0.125 → x = 13 := by
  sorry

end first_divisor_problem_l3123_312306


namespace gcd_of_powers_minus_one_l3123_312386

theorem gcd_of_powers_minus_one (a m n : ℕ) :
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 :=
by sorry

end gcd_of_powers_minus_one_l3123_312386


namespace number_problem_l3123_312360

theorem number_problem (x : ℝ) : 0.6667 * x + 0.75 = 1.6667 → x = 1.375 := by
  sorry

end number_problem_l3123_312360


namespace mixed_tea_sale_price_l3123_312375

/-- Calculates the sale price of mixed tea to earn a specified profit -/
theorem mixed_tea_sale_price
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (profit_percentage : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : profit_percentage = 20)
  : ∃ (sale_price : ℝ), sale_price = 19.2 := by
  sorry

#check mixed_tea_sale_price

end mixed_tea_sale_price_l3123_312375


namespace binomial_prob_three_l3123_312381

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  prob : ℝ → ℝ

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem binomial_prob_three (ξ : BinomialRV 5 (1/3)) :
  ξ.prob 3 = 40/243 := by sorry

end binomial_prob_three_l3123_312381


namespace base_8_246_to_base_10_l3123_312358

def base_8_to_10 (a b c : ℕ) : ℕ := a * 8^2 + b * 8^1 + c * 8^0

theorem base_8_246_to_base_10 : base_8_to_10 2 4 6 = 166 := by
  sorry

end base_8_246_to_base_10_l3123_312358


namespace train_crossing_time_l3123_312315

/-- Prove that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 200 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
  sorry

end train_crossing_time_l3123_312315


namespace exponential_monotonicity_l3123_312362

theorem exponential_monotonicity (a b : ℝ) : a < b → (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  sorry

end exponential_monotonicity_l3123_312362


namespace diophantine_equation_solutions_l3123_312316

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 2^x * 3^y + 1 = 7^z →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2)) :=
by sorry

end diophantine_equation_solutions_l3123_312316


namespace solution_set_when_a_is_one_condition_for_f_geq_g_l3123_312359

-- Define f(x) and g(x)
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x - 1|
def g : ℝ → ℝ := λ x => 2

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ g x} = {x : ℝ | x ≤ 2/3} := by sorry

-- Theorem 2: Condition for f(x) ≥ g(x) to always hold
theorem condition_for_f_geq_g (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g x) ↔ a ≥ 2 := by sorry

end solution_set_when_a_is_one_condition_for_f_geq_g_l3123_312359


namespace oil_container_distribution_l3123_312368

theorem oil_container_distribution :
  ∃ (n m k : ℕ),
    n + m + k = 100 ∧
    n + 10 * m + 50 * k = 500 ∧
    n = 60 ∧ m = 39 ∧ k = 1 := by
  sorry

end oil_container_distribution_l3123_312368


namespace two_mono_triangles_probability_l3123_312397

/-- A complete graph K6 with edges colored either green or yellow -/
structure ColoredK6 where
  edges : Fin 15 → Bool  -- True for green, False for yellow

/-- The probability of an edge being green -/
def p_green : ℚ := 2/3

/-- The probability of an edge being yellow -/
def p_yellow : ℚ := 1/3

/-- The probability of a specific triangle being monochromatic -/
def p_mono_triangle : ℚ := 1/3

/-- The total number of triangles in K6 -/
def total_triangles : ℕ := 20

/-- The probability of exactly two monochromatic triangles in a ColoredK6 -/
def prob_two_mono_triangles : ℚ := 49807360/3486784401

theorem two_mono_triangles_probability (g : ColoredK6) : 
  prob_two_mono_triangles = (total_triangles.choose 2 : ℚ) * p_mono_triangle^2 * (1 - p_mono_triangle)^(total_triangles - 2) :=
sorry

end two_mono_triangles_probability_l3123_312397


namespace angle_ENG_is_45_degrees_l3123_312308

-- Define the rectangle EFGH
structure Rectangle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

-- Define the properties of the rectangle
def is_valid_rectangle (rect : Rectangle) : Prop :=
  rect.E.1 = 0 ∧ rect.E.2 = 0 ∧
  rect.F.1 = 8 ∧ rect.F.2 = 0 ∧
  rect.G.1 = 8 ∧ rect.G.2 = 4 ∧
  rect.H.1 = 0 ∧ rect.H.2 = 4

-- Define point N on side EF
def N : ℝ × ℝ := (4, 0)

-- Define the property that triangle ENG is isosceles
def is_isosceles_ENG (rect : Rectangle) : Prop :=
  let EN := ((N.1 - rect.E.1)^2 + (N.2 - rect.E.2)^2).sqrt
  let NG := ((rect.G.1 - N.1)^2 + (rect.G.2 - N.2)^2).sqrt
  EN = NG

-- Theorem statement
theorem angle_ENG_is_45_degrees (rect : Rectangle) 
  (h1 : is_valid_rectangle rect) 
  (h2 : is_isosceles_ENG rect) : 
  Real.arctan 1 = 45 * (π / 180) :=
sorry

end angle_ENG_is_45_degrees_l3123_312308


namespace only_valid_k_values_l3123_312325

/-- Represents a line in the form y = kx + b --/
structure Line where
  k : ℤ
  b : ℤ

/-- Represents a parabola in the form y = a(x - c)² --/
structure Parabola where
  a : ℤ
  c : ℤ

/-- Checks if a given k value satisfies all conditions --/
def is_valid_k (k : ℤ) : Prop :=
  ∃ (b : ℤ) (a c : ℤ),
    -- Line passes through (-1, 2020)
    2020 = -k + b ∧
    -- Parabola vertex is on the line
    c = -1 - 2020 / k ∧
    -- a is an integer
    a = k^2 / (2020 + k) ∧
    -- k is negative
    k < 0

/-- The main theorem stating that only -404 and -1010 are valid k values --/
theorem only_valid_k_values :
  ∀ k : ℤ, is_valid_k k ↔ k = -404 ∨ k = -1010 := by sorry

end only_valid_k_values_l3123_312325


namespace x_depends_on_m_and_n_l3123_312338

theorem x_depends_on_m_and_n (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃ (a b : ℝ → ℝ → ℝ), ∀ (x : ℝ),
    (x = a m n * m + b m n * n) →
    ((x + m)^3 - (x + n)^3 = (m - n)^3) →
    (a m n ≠ 1 ∨ b m n ≠ 1) ∧
    (a m n ≠ -1 ∨ b m n ≠ 1) ∧
    (a m n ≠ 1 ∨ b m n ≠ -1) ∧
    (a m n ≠ -1 ∨ b m n ≠ -1) :=
by sorry

end x_depends_on_m_and_n_l3123_312338


namespace part1_part2_l3123_312346

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the set A
def A (a b c : ℝ) : Set ℝ := {x | f a b c x = x}

-- Part 1
theorem part1 (a b c : ℝ) :
  A a b c = {1, 2} → f a b c 0 = 2 →
  ∃ (M m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x ≤ M ∧ m ≤ f a b c x) ∧ M = 10 ∧ m = 1 :=
sorry

-- Part 2
theorem part2 (a b c : ℝ) :
  A a b c = {2} → a ≥ 1 →
  ∃ (g : ℝ → ℝ), (∀ a' ≥ 1, g a' ≥ 63/4) ∧
    (∃ (M m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x ≤ M ∧ m ≤ f a b c x) ∧ g a = M + m) :=
sorry

end part1_part2_l3123_312346


namespace hillary_stop_distance_l3123_312361

/-- Proves that Hillary stops 2900 feet short of the summit given the climbing conditions --/
theorem hillary_stop_distance (summit_distance : ℝ) (hillary_rate : ℝ) (eddy_rate : ℝ) 
  (hillary_descent_rate : ℝ) (climb_time : ℝ) 
  (h1 : summit_distance = 4700)
  (h2 : hillary_rate = 800)
  (h3 : eddy_rate = 500)
  (h4 : hillary_descent_rate = 1000)
  (h5 : climb_time = 6) :
  ∃ x : ℝ, x = 2900 ∧ 
  (summit_distance - x) + (eddy_rate * climb_time) + x = 
  summit_distance + (hillary_rate * climb_time - (summit_distance - x)) :=
by sorry

end hillary_stop_distance_l3123_312361


namespace expression_evaluation_l3123_312313

theorem expression_evaluation :
  let x : ℝ := 2
  (x + 1) * (x - 1) + x * (3 - x) = 5 := by
sorry

end expression_evaluation_l3123_312313


namespace elidas_name_length_l3123_312391

theorem elidas_name_length :
  ∀ (E A : ℕ),
  A = 2 * E - 2 →
  10 * ((E + A) / 2 : ℚ) = 65 →
  E = 5 :=
by sorry

end elidas_name_length_l3123_312391


namespace afternoon_and_evening_emails_l3123_312327

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- Theorem: The sum of emails Jack received in the afternoon and evening is 13 -/
theorem afternoon_and_evening_emails :
  afternoon_emails + evening_emails = 13 := by sorry

end afternoon_and_evening_emails_l3123_312327


namespace remainder_x_power_10_minus_1_div_x_plus_1_l3123_312356

theorem remainder_x_power_10_minus_1_div_x_plus_1 (x : ℝ) : 
  (x^10 - 1) % (x + 1) = 0 := by
sorry

end remainder_x_power_10_minus_1_div_x_plus_1_l3123_312356


namespace cistern_filling_time_l3123_312303

theorem cistern_filling_time (p q : ℝ) (h1 : p > 0) (h2 : q > 0) : 
  p = 1 / 12 → q = 1 / 15 → 
  let combined_rate := p + q
  let filled_portion := 4 * combined_rate
  let remaining_portion := 1 - filled_portion
  remaining_portion / q = 6 := by sorry

end cistern_filling_time_l3123_312303


namespace unique_element_in_A_not_in_B_l3123_312309

-- Define the sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {2, 4, 6}

-- State the theorem
theorem unique_element_in_A_not_in_B :
  ∀ x : ℕ, x ∈ A ∧ x ∉ B → x = 3 := by
  sorry

end unique_element_in_A_not_in_B_l3123_312309


namespace green_tiles_in_50th_row_l3123_312396

/-- Represents the number of tiles in a row of the tiling pattern. -/
def num_tiles (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of green tiles in a row of the tiling pattern. -/
def num_green_tiles (n : ℕ) : ℕ := (num_tiles n - 1) / 2

theorem green_tiles_in_50th_row :
  num_green_tiles 50 = 49 := by sorry

end green_tiles_in_50th_row_l3123_312396


namespace max_value_trigonometric_function_l3123_312323

theorem max_value_trigonometric_function :
  ∀ θ : ℝ, 0 < θ → θ < π / 2 →
  (∀ φ : ℝ, 0 < φ → φ < π / 2 →
    (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) ≥ (1 / Real.sin φ - 1) * (1 / Real.cos φ - 1)) →
  (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) = 3 - 2 * Real.sqrt 2 :=
by sorry


end max_value_trigonometric_function_l3123_312323


namespace intersection_implies_t_equals_two_l3123_312300

theorem intersection_implies_t_equals_two (t : ℝ) : 
  let M : Set ℝ := {1, t^2}
  let N : Set ℝ := {-2, t+2}
  (M ∩ N).Nonempty → t = 2 := by
sorry

end intersection_implies_t_equals_two_l3123_312300


namespace union_equality_iff_a_in_range_l3123_312350

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x ≤ x - a}
def B : Set ℝ := {x | 4*x - x^2 - 3 ≥ 0}

-- State the theorem
theorem union_equality_iff_a_in_range : 
  ∀ a : ℝ, (A a ∪ B = B) ↔ a ∈ Set.Icc 1 3 := by sorry

end union_equality_iff_a_in_range_l3123_312350


namespace exists_polynomial_with_negative_coeff_positive_powers_l3123_312345

theorem exists_polynomial_with_negative_coeff_positive_powers :
  ∃ (P : Polynomial ℝ), 
    (∃ (i : ℕ), (P.coeff i) < 0) ∧ 
    (∀ (n : ℕ), n > 1 → ∀ (j : ℕ), ((P ^ n).coeff j) > 0) := by
  sorry

end exists_polynomial_with_negative_coeff_positive_powers_l3123_312345


namespace problem_statement_l3123_312342

theorem problem_statement (a b : ℝ) : (2*a + b)^2 + |b - 2| = 0 → (-a - b)^2014 = 1 := by
  sorry

end problem_statement_l3123_312342


namespace square_floor_with_57_black_tiles_has_841_total_tiles_l3123_312385

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor where
  side_length : ℕ
  is_valid : side_length > 0

/-- Calculates the number of black tiles on the diagonals of a square floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Calculates the total number of tiles on a square floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem stating that a square floor with 57 black tiles on its diagonals has 841 total tiles. -/
theorem square_floor_with_57_black_tiles_has_841_total_tiles :
  ∀ (floor : SquareFloor), black_tiles floor = 57 → total_tiles floor = 841 :=
by
  sorry

end square_floor_with_57_black_tiles_has_841_total_tiles_l3123_312385


namespace rectangle_perimeter_l3123_312372

theorem rectangle_perimeter (area : ℝ) (side : ℝ) (h1 : area = 108) (h2 : side = 12) :
  2 * (side + area / side) = 42 := by
  sorry

end rectangle_perimeter_l3123_312372


namespace total_cards_l3123_312329

/-- The number of cards each person has -/
structure Cards where
  janet : ℕ
  brenda : ℕ
  mara : ℕ

/-- The conditions of the problem -/
def problem_conditions (c : Cards) : Prop :=
  c.janet = c.brenda + 9 ∧
  c.mara = 2 * c.janet ∧
  c.mara = 150 - 40

/-- The theorem to prove -/
theorem total_cards (c : Cards) (h : problem_conditions c) : 
  c.janet + c.brenda + c.mara = 211 := by
  sorry

end total_cards_l3123_312329


namespace first_stack_height_is_correct_l3123_312370

/-- The height of the first stack of blocks -/
def first_stack_height : ℕ := 7

/-- The height of the second stack of blocks -/
def second_stack_height : ℕ := first_stack_height + 5

/-- The height of the third stack of blocks -/
def third_stack_height : ℕ := second_stack_height + 7

/-- The number of blocks that fell from the second stack -/
def fallen_second_stack : ℕ := second_stack_height - 2

/-- The number of blocks that fell from the third stack -/
def fallen_third_stack : ℕ := third_stack_height - 3

/-- The total number of fallen blocks -/
def total_fallen_blocks : ℕ := 33

theorem first_stack_height_is_correct :
  first_stack_height + fallen_second_stack + fallen_third_stack = total_fallen_blocks :=
by sorry

end first_stack_height_is_correct_l3123_312370


namespace compute_expression_l3123_312319

theorem compute_expression : 12 + 4 * (5 - 10)^3 = -488 := by
  sorry

end compute_expression_l3123_312319


namespace difference_of_squares_division_l3123_312365

theorem difference_of_squares_division : (245^2 - 225^2) / 20 = 470 := by
  sorry

end difference_of_squares_division_l3123_312365


namespace quadratic_rewrite_l3123_312371

theorem quadratic_rewrite (k : ℝ) :
  ∃ (d r s : ℝ), 9 * k^2 - 6 * k + 15 = d * (k + r)^2 + s ∧ s / r = -42 := by
  sorry

end quadratic_rewrite_l3123_312371


namespace geometric_sequence_sum_l3123_312347

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
sorry

end geometric_sequence_sum_l3123_312347


namespace cone_sphere_ratio_l3123_312349

theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (R : ℝ) : 
  R = 2 * r →
  (1/3) * (4/3) * Real.pi * r^3 = (1/3) * Real.pi * R^2 * h →
  h / R = 1/6 :=
by sorry

end cone_sphere_ratio_l3123_312349


namespace f_even_and_increasing_l3123_312363

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end f_even_and_increasing_l3123_312363


namespace students_not_in_same_row_or_column_l3123_312353

/-- Represents a student's position in a classroom --/
structure Position where
  row : ℕ
  column : ℕ

/-- Defines the seating arrangement for students A and B --/
def seating_arrangement : (Position × Position) :=
  (⟨3, 6⟩, ⟨6, 3⟩)

/-- Theorem stating that students A and B are not in the same row or column --/
theorem students_not_in_same_row_or_column :
  let (student_a, student_b) := seating_arrangement
  (student_a.row ≠ student_b.row) ∧ (student_a.column ≠ student_b.column) := by
  sorry

#check students_not_in_same_row_or_column

end students_not_in_same_row_or_column_l3123_312353


namespace gcd_1987_1463_l3123_312326

theorem gcd_1987_1463 : Nat.gcd 1987 1463 = 1 := by
  sorry

end gcd_1987_1463_l3123_312326


namespace circle_area_decrease_l3123_312355

theorem circle_area_decrease (r : ℝ) (h : r > 0) :
  let new_r := r / 2
  let original_area := π * r^2
  let new_area := π * new_r^2
  (original_area - new_area) / original_area = 3/4 := by sorry

end circle_area_decrease_l3123_312355


namespace max_value_of_expression_l3123_312383

def S : Set ℝ := {1, 2, 3, 5, 10}

theorem max_value_of_expression (x y : ℝ) (hx : x ∈ S) (hy : y ∈ S) :
  (x / y + y / x) ≤ 10.1 :=
sorry

end max_value_of_expression_l3123_312383


namespace abs_ab_value_l3123_312366

/-- Given an ellipse and a hyperbola with specific foci, prove that |ab| = 2√65 -/
theorem abs_ab_value (a b : ℝ) : 
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 4) ∨ (x = 0 ∧ y = -4)) →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 65 := by
  sorry

end abs_ab_value_l3123_312366


namespace function_inequality_implies_a_bound_l3123_312335

open Real

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, x^2 * exp x > 3 * exp x + a) →
  a < exp 2 := by
  sorry

end function_inequality_implies_a_bound_l3123_312335


namespace bees_count_l3123_312305

theorem bees_count (first_day_count : ℕ) (second_day_multiplier : ℕ) : first_day_count = 144 → second_day_multiplier = 3 → first_day_count * second_day_multiplier = 432 := by
  sorry

end bees_count_l3123_312305


namespace expression_value_l3123_312364

theorem expression_value : (2018 - 18 + 20) / 2 = 1010 := by
  sorry

end expression_value_l3123_312364


namespace right_triangle_leg_length_l3123_312314

theorem right_triangle_leg_length (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 1 →
  4 * (1/2 * triangle_leg * triangle_leg) = square_side * square_side →
  triangle_leg = Real.sqrt 2 / 2 := by
  sorry

end right_triangle_leg_length_l3123_312314


namespace sunflower_seed_contest_l3123_312310

theorem sunflower_seed_contest (player1 player2 player3 player4 total : ℕ) :
  player1 = 78 →
  player2 = 53 →
  player3 = player2 + 30 →
  player4 = 2 * player3 →
  total = player1 + player2 + player3 + player4 →
  total = 380 := by
  sorry

end sunflower_seed_contest_l3123_312310


namespace min_value_in_region_D_l3123_312387

def region_D (x y : ℝ) : Prop :=
  y ≤ x ∧ y ≥ -x ∧ x ≤ (Real.sqrt 2) / 2

def objective_function (x y : ℝ) : ℝ :=
  x - 2 * y

theorem min_value_in_region_D :
  ∃ (min : ℝ), min = -(Real.sqrt 2) / 2 ∧
  ∀ (x y : ℝ), region_D x y → objective_function x y ≥ min :=
sorry

end min_value_in_region_D_l3123_312387


namespace fifth_term_is_32_l3123_312376

/-- A sequence where the difference between each term and its predecessor increases by 3 each time -/
def special_sequence : ℕ → ℕ
| 0 => 2
| 1 => 5
| n + 2 => special_sequence (n + 1) + 3 * (n + 1)

theorem fifth_term_is_32 : special_sequence 4 = 32 := by
  sorry

end fifth_term_is_32_l3123_312376


namespace all_propositions_false_l3123_312392

-- Define the basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the geometric relations
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry

-- State the propositions
def proposition1 : Prop :=
  ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2

def proposition2 : Prop :=
  ∀ p1 p2 p3 : Plane, perpendicular_to_plane p1 p3 → perpendicular_to_plane p2 p3 → parallel_planes p1 p2

def proposition3 : Prop :=
  ∀ l1 l2 : Line, ∀ p : Plane, angle_with_plane l1 p = angle_with_plane l2 p → parallel l1 l2

def proposition4 : Prop :=
  ∀ l1 l2 l3 l4 : Line, skew l1 l2 → intersect l3 l1 → intersect l3 l2 → intersect l4 l1 → intersect l4 l2 → skew l3 l4

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by sorry

end all_propositions_false_l3123_312392


namespace sum_of_squares_l3123_312337

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 116) 
  (h2 : a + b + c = 22) : 
  a^2 + b^2 + c^2 = 252 := by
sorry

end sum_of_squares_l3123_312337
