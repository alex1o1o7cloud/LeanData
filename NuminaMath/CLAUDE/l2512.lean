import Mathlib

namespace point_B_coordinates_l2512_251263

-- Define the coordinates of points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- Theorem statement
theorem point_B_coordinates :
  ∀ a : ℝ, (A a).1 = 0 → B a = (4, -4) := by
  sorry

end point_B_coordinates_l2512_251263


namespace specific_coin_expected_value_l2512_251221

/-- A biased coin with probabilities for heads and tails, and associated winnings/losses. -/
structure BiasedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  win_heads : ℚ
  loss_tails : ℚ

/-- Expected value of winnings for a single flip of a biased coin. -/
def expected_value (coin : BiasedCoin) : ℚ :=
  coin.prob_heads * coin.win_heads + coin.prob_tails * (-coin.loss_tails)

/-- Theorem stating the expected value for the specific coin in the problem. -/
theorem specific_coin_expected_value :
  let coin : BiasedCoin := {
    prob_heads := 1/4,
    prob_tails := 3/4,
    win_heads := 4,
    loss_tails := 3
  }
  expected_value coin = -5/4 := by sorry

end specific_coin_expected_value_l2512_251221


namespace sum_of_xyz_is_twelve_l2512_251278

theorem sum_of_xyz_is_twelve (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x * y = x + y) (hyz : y * z = 3 * (y + z)) (hzx : z * x = 2 * (z + x)) :
  x + y + z = 12 := by
  sorry

end sum_of_xyz_is_twelve_l2512_251278


namespace range_of_y_l2512_251250

theorem range_of_y (x y : ℝ) (h1 : x = 4 - y) (h2 : -2 ≤ x ∧ x ≤ -1) : 5 ≤ y ∧ y ≤ 6 := by
  sorry

end range_of_y_l2512_251250


namespace max_rational_products_l2512_251219

/-- Represents a number that can be either rational or irrational -/
inductive Number
| Rational : ℚ → Number
| Irrational : ℝ → Number

/-- Definition of the table structure -/
structure Table :=
  (top : Fin 50 → Number)
  (left : Fin 50 → Number)

/-- Counts the number of rational and irrational numbers in a list -/
def countNumbers (numbers : List Number) : Nat × Nat :=
  numbers.foldl (fun (ratCount, irratCount) n =>
    match n with
    | Number.Rational _ => (ratCount + 1, irratCount)
    | Number.Irrational _ => (ratCount, irratCount + 1)
  ) (0, 0)

/-- Checks if the product of two Numbers is rational -/
def isRationalProduct (a b : Number) : Bool :=
  match a, b with
  | Number.Rational _, Number.Rational _ => true
  | Number.Rational 0, _ => true
  | _, Number.Rational 0 => true
  | _, _ => false

/-- Counts the number of rational products in the table -/
def countRationalProducts (t : Table) : Nat :=
  (List.range 50).foldl (fun count i =>
    (List.range 50).foldl (fun count j =>
      if isRationalProduct (t.top i) (t.left j) then count + 1 else count
    ) count
  ) 0

/-- The main theorem -/
theorem max_rational_products (t : Table) :
  (countNumbers (List.ofFn t.top) = (25, 25) ∧
   countNumbers (List.ofFn t.left) = (25, 25) ∧
   (∀ i j : Fin 50, t.top i ≠ t.left j) ∧
   (∃ i : Fin 50, t.top i = Number.Rational 0)) →
  countRationalProducts t ≤ 1275 :=
sorry

end max_rational_products_l2512_251219


namespace glutenNutNonVegan_is_65_l2512_251286

/-- Represents the number of cupcakes with specific properties -/
structure Cupcakes where
  total : ℕ
  glutenFree : ℕ
  vegan : ℕ
  nutFree : ℕ
  glutenFreeVegan : ℕ
  veganNutFree : ℕ

/-- The properties of the cupcakes ordered for the birthday party -/
def birthdayCupcakes : Cupcakes where
  total := 120
  glutenFree := 120 / 3
  vegan := 120 / 4
  nutFree := 120 / 5
  glutenFreeVegan := 15
  veganNutFree := 10

/-- Calculates the number of cupcakes that are gluten, nut, and non-vegan -/
def glutenNutNonVegan (c : Cupcakes) : ℕ :=
  c.total - (c.glutenFree + (c.vegan - c.glutenFreeVegan))

/-- Theorem stating that the number of gluten, nut, and non-vegan cupcakes is 65 -/
theorem glutenNutNonVegan_is_65 : glutenNutNonVegan birthdayCupcakes = 65 := by
  sorry

end glutenNutNonVegan_is_65_l2512_251286


namespace cycle_original_price_l2512_251224

/-- Given a cycle sold at a 12% loss for Rs. 1408, prove that the original price was Rs. 1600. -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1408 → 
  loss_percentage = 12 → 
  (1 - loss_percentage / 100) * 1600 = selling_price :=
by sorry

end cycle_original_price_l2512_251224


namespace inscribed_rectangle_area_l2512_251254

/-- Given a right triangle ABC with vertices A(45,0), B(20,0), and C(0,30),
    and an inscribed rectangle DEFG where the area of triangle CGF is 351,
    prove that the area of rectangle DEFG is 468. -/
theorem inscribed_rectangle_area (A B C D E F G : ℝ × ℝ) : 
  A = (45, 0) →
  B = (20, 0) →
  C = (0, 30) →
  (D.1 ≥ 0 ∧ D.1 ≤ 45 ∧ D.2 = 0) →
  (E.1 = D.1 ∧ E.2 > 0 ∧ E.2 < 30) →
  (F.1 = 20 ∧ F.2 = E.2) →
  (G.1 = 0 ∧ G.2 = E.2) →
  (C.2 - E.2) * F.1 / 2 = 351 →
  (F.1 - D.1) * E.2 = 468 :=
by sorry

end inscribed_rectangle_area_l2512_251254


namespace line_through_intersection_and_perpendicular_l2512_251258

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 7 * x - 8 * y - 1 = 0
def l2 (x y : ℝ) : Prop := 2 * x + 17 * y + 9 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x - y + 7 = 0

-- Define the resulting line
def result_line (x y : ℝ) : Prop := 27 * x + 54 * y + 37 = 0

-- Theorem statement
theorem line_through_intersection_and_perpendicular :
  ∃ (x y : ℝ),
    (l1 x y ∧ l2 x y) ∧  -- Intersection point satisfies both l1 and l2
    (∀ (x' y' : ℝ), result_line x' y' ↔ 
      (y' - y) = -(1/2) * (x' - x)) ∧  -- Slope of result_line is -1/2
    (∀ (x' y' : ℝ), perp_line x' y' → 
      (y' - y) * (1/2) + (x' - x) = 0)  -- Perpendicular to perp_line
    := by sorry

end line_through_intersection_and_perpendicular_l2512_251258


namespace pepperoni_to_crust_ratio_is_one_to_three_l2512_251299

/-- Represents the calorie content of various food items and portions consumed --/
structure FoodCalories where
  lettuce : ℕ
  carrot : ℕ
  dressing : ℕ
  crust : ℕ
  cheese : ℕ
  saladPortion : ℚ
  pizzaPortion : ℚ
  totalConsumed : ℕ

/-- Calculates the ratio of pepperoni calories to crust calories --/
def pepperoniToCrustRatio (food : FoodCalories) : ℚ × ℚ :=
  sorry

/-- Theorem stating that given the conditions, the ratio of pepperoni to crust calories is 1:3 --/
theorem pepperoni_to_crust_ratio_is_one_to_three 
  (food : FoodCalories)
  (h1 : food.lettuce = 50)
  (h2 : food.carrot = 2 * food.lettuce)
  (h3 : food.dressing = 210)
  (h4 : food.crust = 600)
  (h5 : food.cheese = 400)
  (h6 : food.saladPortion = 1/4)
  (h7 : food.pizzaPortion = 1/5)
  (h8 : food.totalConsumed = 330) :
  pepperoniToCrustRatio food = (1, 3) :=
sorry

end pepperoni_to_crust_ratio_is_one_to_three_l2512_251299


namespace rotation_of_vector_l2512_251259

/-- Given points O and P in a 2D Cartesian plane, and Q obtained by rotating OP counterclockwise by 3π/4, 
    prove that Q has the specified coordinates. -/
theorem rotation_of_vector (O P Q : ℝ × ℝ) : 
  O = (0, 0) → 
  P = (6, 8) → 
  Q = (Real.cos (3 * Real.pi / 4) * (P.1 - O.1) - Real.sin (3 * Real.pi / 4) * (P.2 - O.2) + O.1,
       Real.sin (3 * Real.pi / 4) * (P.1 - O.1) + Real.cos (3 * Real.pi / 4) * (P.2 - O.2) + O.2) →
  Q = (-7 * Real.sqrt 2, -Real.sqrt 2) :=
by sorry

end rotation_of_vector_l2512_251259


namespace max_product_constrained_max_product_achieved_l2512_251237

theorem max_product_constrained (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 20 → x * y ≤ 25 := by
  sorry

theorem max_product_achieved (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 20 → x = 10 ∧ y = 2.5 → x * y = 25 := by
  sorry

end max_product_constrained_max_product_achieved_l2512_251237


namespace grandfather_is_73_l2512_251241

/-- Xiaowen's age in years -/
def xiaowens_age : ℕ := 13

/-- Xiaowen's grandfather's age in years -/
def grandfathers_age : ℕ := 5 * xiaowens_age + 8

/-- Theorem stating that Xiaowen's grandfather is 73 years old -/
theorem grandfather_is_73 : grandfathers_age = 73 := by
  sorry

end grandfather_is_73_l2512_251241


namespace f_minimum_value_l2512_251202

def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

theorem f_minimum_value : ∀ x : ℕ+, f x ≥ 23/2 := by sorry

end f_minimum_value_l2512_251202


namespace cylinder_volume_square_cross_section_l2512_251253

/-- The volume of a cylinder with a square cross-section of area 4 is 2π. -/
theorem cylinder_volume_square_cross_section (a : ℝ) (h : a = 4) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π * 2 = 2 * π := by
  sorry

end cylinder_volume_square_cross_section_l2512_251253


namespace solution_set_inequality_l2512_251238

theorem solution_set_inequality (x : ℝ) :
  x^2 * (x - 4) ≥ 0 ↔ x ≥ 4 ∨ x = 0 := by sorry

end solution_set_inequality_l2512_251238


namespace triangle_angle_A_l2512_251292

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hC : 0 < B) :
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = π / 3 →
  ∃ (A : ℝ), 0 < A ∧ A < 2 * π / 3 ∧ Real.sin A = Real.sqrt 2 / 2 ∧ A = π / 4 :=
by sorry

end triangle_angle_A_l2512_251292


namespace mutually_exclusive_necessary_not_sufficient_l2512_251276

open Set

variable {Ω : Type*} [MeasurableSpace Ω]

def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∪ A₂ = univ ∧ A₁ ∩ A₂ = ∅

theorem mutually_exclusive_necessary_not_sufficient :
  (∀ A₁ A₂ : Set Ω, complementary A₁ A₂ → mutually_exclusive A₁ A₂) ∧
  (∃ A₁ A₂ : Set Ω, mutually_exclusive A₁ A₂ ∧ ¬complementary A₁ A₂) :=
sorry

end mutually_exclusive_necessary_not_sufficient_l2512_251276


namespace perpendicular_line_x_intercept_l2512_251288

/-- Given a line L1 defined by 4x + 5y = 20, and a perpendicular line L2 with y-intercept -3,
    the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x + 5 * y = 20
  let m1 : ℝ := -4 / 5  -- slope of L1
  let m2 : ℝ := 5 / 4   -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y => y = m2 * x - 3  -- equation of L2
  ∃ x : ℝ, L2 x 0 ∧ x = 12 / 5
  :=
by sorry

end perpendicular_line_x_intercept_l2512_251288


namespace max_value_inequality_l2512_251295

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  (a * (b + 2 * c)) ^ (1/4) + (b * (c + 2 * d)) ^ (1/4) + 
  (c * (d + 2 * a)) ^ (1/4) + (d * (a + 2 * b)) ^ (1/4) ≤ 4 * 3 ^ (1/4) := by
  sorry

end max_value_inequality_l2512_251295


namespace stewart_farm_sheep_count_l2512_251262

/-- The number of sheep on the Stewart farm -/
def num_sheep : ℕ := 24

/-- The number of horses on the Stewart farm -/
def num_horses : ℕ := 56

/-- The ratio of sheep to horses -/
def sheep_to_horse_ratio : ℚ := 3 / 7

/-- The amount of food each horse eats per day in ounces -/
def horse_food_per_day : ℕ := 230

/-- The total amount of horse food needed per day in ounces -/
def total_horse_food_per_day : ℕ := 12880

theorem stewart_farm_sheep_count :
  (num_sheep : ℚ) / num_horses = sheep_to_horse_ratio ∧
  num_horses * horse_food_per_day = total_horse_food_per_day ∧
  num_sheep = 24 := by
  sorry

end stewart_farm_sheep_count_l2512_251262


namespace ball_probability_6_l2512_251270

def ball_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k + 2 => (1 / 3) * (1 - ball_probability (k + 1))

theorem ball_probability_6 :
  ball_probability 6 = 61 / 243 := by
  sorry

end ball_probability_6_l2512_251270


namespace total_money_l2512_251234

theorem total_money (mark_money : Rat) (carolyn_money : Rat) (jack_money : Rat)
  (h1 : mark_money = 4 / 5)
  (h2 : carolyn_money = 2 / 5)
  (h3 : jack_money = 1 / 2) :
  mark_money + carolyn_money + jack_money = 17 / 10 := by
  sorry

end total_money_l2512_251234


namespace cricket_average_increase_l2512_251236

theorem cricket_average_increase (initial_average : ℝ) : 
  (16 * initial_average + 92) / 17 = initial_average + 4 → 
  (16 * initial_average + 92) / 17 = 28 := by
sorry

end cricket_average_increase_l2512_251236


namespace arithmetic_sequence_3_2_3_3_l2512_251228

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term
  is_arithmetic : b - a = c - b

/-- The second term of an arithmetic sequence with 3^2 as first term and 3^3 as third term is 18 -/
theorem arithmetic_sequence_3_2_3_3 :
  ∃ (seq : ArithmeticSequence3), seq.a = 3^2 ∧ seq.c = 3^3 ∧ seq.b = 18 := by
  sorry

end arithmetic_sequence_3_2_3_3_l2512_251228


namespace quadratic_roots_difference_l2512_251203

theorem quadratic_roots_difference (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) → 
  a > b → 
  a - b = 8 := by sorry

end quadratic_roots_difference_l2512_251203


namespace rectangle_containment_l2512_251235

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℕ+
  height : ℕ+

/-- The set of all rectangles -/
def RectangleSet : Set Rectangle := {r : Rectangle | True}

/-- One rectangle is contained within another -/
def contained (r1 r2 : Rectangle) : Prop :=
  r1.width ≤ r2.width ∧ r1.height ≤ r2.height

theorem rectangle_containment (h : Set.Infinite RectangleSet) :
  ∃ (r1 r2 : Rectangle), r1 ∈ RectangleSet ∧ r2 ∈ RectangleSet ∧ contained r1 r2 :=
sorry

end rectangle_containment_l2512_251235


namespace simplify_and_evaluate_l2512_251272

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4*x - 4 = 0) :
  3*(x-2)^2 - 6*(x+1)*(x-1) = 6 := by sorry

end simplify_and_evaluate_l2512_251272


namespace perimeter_after_adding_tiles_l2512_251239

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- The initial "T" shaped configuration -/
def initial_config : TileConfiguration :=
  { tiles := 6, perimeter := 12 }

/-- The number of tiles added -/
def added_tiles : ℕ := 3

/-- A function that calculates the new perimeter after adding tiles -/
def new_perimeter (config : TileConfiguration) (added : ℕ) : ℕ :=
  sorry

theorem perimeter_after_adding_tiles :
  new_perimeter initial_config added_tiles = 16 := by
  sorry

end perimeter_after_adding_tiles_l2512_251239


namespace amp_fifteen_amp_l2512_251274

-- Define the & operation
def amp (x : ℝ) : ℝ := 9 - x

-- Define the & prefix operation
def amp_prefix (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_fifteen_amp : amp_prefix (amp 15) = -15 := by sorry

end amp_fifteen_amp_l2512_251274


namespace integer_solutions_system_l2512_251225

theorem integer_solutions_system (x y z t : ℤ) : 
  (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
  ((x, y, z, t) = (1, 0, 3, 1) ∨ 
   (x, y, z, t) = (-1, 0, -3, -1) ∨ 
   (x, y, z, t) = (3, 1, 1, 0) ∨ 
   (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end integer_solutions_system_l2512_251225


namespace sum_15_with_9_dice_l2512_251289

/-- The number of ways to distribute n indistinguishable objects among k distinct containers,
    with no container receiving more than m objects. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to throw 9 fair 6-sided dice such that their sum is 15. -/
def ways_to_sum_15 : ℕ := distribute 6 9 5

theorem sum_15_with_9_dice : ways_to_sum_15 = 3003 := by sorry

end sum_15_with_9_dice_l2512_251289


namespace floor_e_equals_two_l2512_251264

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_equals_two_l2512_251264


namespace f_monotone_increasing_intervals_l2512_251291

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem f_monotone_increasing_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) := by sorry

end f_monotone_increasing_intervals_l2512_251291


namespace rohit_final_position_l2512_251245

/-- Represents a 2D position --/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a direction --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Rohit's movement --/
def move (p : Position) (d : Direction) (distance : ℝ) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + distance⟩
  | Direction.East => ⟨p.x + distance, p.y⟩
  | Direction.South => ⟨p.x, p.y - distance⟩
  | Direction.West => ⟨p.x - distance, p.y⟩

/-- Represents a left turn --/
def turnLeft (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.East => Direction.North
  | Direction.South => Direction.East
  | Direction.West => Direction.South

/-- Represents a right turn --/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

theorem rohit_final_position : 
  let start : Position := ⟨0, 0⟩
  let p1 := move start Direction.South 25
  let d1 := turnLeft Direction.South
  let p2 := move p1 d1 20
  let d2 := turnLeft d1
  let p3 := move p2 d2 25
  let d3 := turnRight d2
  let final := move p3 d3 15
  final = ⟨35, 0⟩ := by sorry

end rohit_final_position_l2512_251245


namespace perpendicular_condition_l2512_251281

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "lies within" relation for a line and a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define non-coincidence for lines
variable (non_coincident : Line → Line → Line → Prop)

theorem perpendicular_condition 
  (l m n : Line) (α : Plane)
  (h_non_coincident : non_coincident l m n)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_α : line_in_plane n α) :
  (perp_line_plane l α → perp_line_line l m ∧ perp_line_line l n) ∧
  ¬(perp_line_line l m ∧ perp_line_line l n → perp_line_plane l α) :=
by sorry

end perpendicular_condition_l2512_251281


namespace ellipse_standard_equation_ellipse_trajectory_equation_l2512_251231

/-- An ellipse with center at origin, left focus at (-√3, 0), right vertex at (2, 0), and point A at (1, 1/2) -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ
  point_A : ℝ × ℝ
  h_center : center = (0, 0)
  h_left_focus : left_focus = (-Real.sqrt 3, 0)
  h_right_vertex : right_vertex = (2, 0)
  h_point_A : point_A = (1, 1/2)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- The trajectory equation of the midpoint M of line segment PA -/
def trajectory_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (2*x - 1)^2 / 4 + (2*y - 1/2)^2 = 1

/-- Theorem stating the standard equation of the ellipse -/
theorem ellipse_standard_equation (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | standard_equation e p.1 p.2} ↔ 
    (x, y) ∈ {p : ℝ × ℝ | x^2 / 4 + y^2 = 1} :=
sorry

/-- Theorem stating the trajectory equation of the midpoint M -/
theorem ellipse_trajectory_equation (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | trajectory_equation e p.1 p.2} ↔ 
    (x, y) ∈ {p : ℝ × ℝ | (2*x - 1)^2 / 4 + (2*y - 1/2)^2 = 1} :=
sorry

end ellipse_standard_equation_ellipse_trajectory_equation_l2512_251231


namespace baguettes_per_batch_is_48_l2512_251283

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := sorry

/-- The number of batches made per day -/
def batches_per_day : ℕ := 3

/-- The number of baguettes sold after the first batch -/
def sold_after_first : ℕ := 37

/-- The number of baguettes sold after the second batch -/
def sold_after_second : ℕ := 52

/-- The number of baguettes sold after the third batch -/
def sold_after_third : ℕ := 49

/-- The number of baguettes left at the end -/
def baguettes_left : ℕ := 6

theorem baguettes_per_batch_is_48 :
  baguettes_per_batch = 48 ∧
  baguettes_per_batch * batches_per_day - 
  (sold_after_first + sold_after_second + sold_after_third) = 
  baguettes_left :=
sorry

end baguettes_per_batch_is_48_l2512_251283


namespace only_equation_II_has_nontrivial_solution_l2512_251255

theorem only_equation_II_has_nontrivial_solution :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (Real.sqrt (a^2 + b^2 + c^2) = c) ∧
  (∀ (x y z : ℝ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) →
    (Real.sqrt (x^2 + y^2 + z^2) ≠ 0) ∧
    (Real.sqrt (x^2 + y^2 + z^2) ≠ x + y + z) ∧
    (Real.sqrt (x^2 + y^2 + z^2) ≠ x*y*z)) :=
by sorry

end only_equation_II_has_nontrivial_solution_l2512_251255


namespace four_digit_number_properties_l2512_251249

/-- P function for a four-digit number -/
def P (x : ℕ) : ℤ :=
  let y := (x % 10) * 1000 + x / 10
  (x - y) / 9

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k * k

/-- Definition of s -/
def s (a b : ℕ) : ℕ := 1100 + 20 * a + b

/-- Definition of t -/
def t (a b : ℕ) : ℕ := b * 1000 + a * 100 + 23

/-- Main theorem -/
theorem four_digit_number_properties :
  (P 5324 = 88) ∧
  (∀ a b : ℕ, 1 ≤ a → a ≤ 4 → 1 ≤ b → b ≤ 9 →
    (∃ min_pt : ℤ, min_pt = -161 ∧
      is_perfect_square (P (t a b) - P (s a b) - a - b) ∧
      (∀ a' b' : ℕ, 1 ≤ a' → a' ≤ 4 → 1 ≤ b' → b' ≤ 9 →
        is_perfect_square (P (t a' b') - P (s a' b') - a' - b') →
        P (t a' b') ≥ min_pt))) :=
sorry

end four_digit_number_properties_l2512_251249


namespace marble_probability_l2512_251279

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
  (h_total : total = 90)
  (h_white : p_white = 1 / 6)
  (h_green : p_green = 1 / 5) :
  1 - (p_white + p_green) = 19 / 30 := by
  sorry

end marble_probability_l2512_251279


namespace marathon_theorem_l2512_251269

def marathon_problem (total_distance : ℝ) (day1_percent : ℝ) (day2_percent : ℝ) : ℝ :=
  let day1_distance := total_distance * day1_percent
  let remaining_after_day1 := total_distance - day1_distance
  let day2_distance := remaining_after_day1 * day2_percent
  let day3_distance := total_distance - day1_distance - day2_distance
  day3_distance

theorem marathon_theorem :
  marathon_problem 70 0.2 0.5 = 28 := by
  sorry

end marathon_theorem_l2512_251269


namespace intersection_of_P_and_Q_l2512_251243

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}
def Q : Set ℝ := {x | ∃ y : ℝ, y = Real.log (2 - x)}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 1 2 := by sorry

end intersection_of_P_and_Q_l2512_251243


namespace max_quarters_is_twelve_l2512_251293

/-- Represents the number of each type of coin -/
structure CoinCount where
  count : ℕ

/-- Represents the total value of coins in cents -/
def total_value (c : CoinCount) : ℕ :=
  25 * c.count + 5 * c.count + 10 * c.count

/-- The maximum number of quarters possible given the conditions -/
def max_quarters : Prop :=
  ∃ (c : CoinCount), total_value c = 480 ∧ 
    ∀ (c' : CoinCount), total_value c' = 480 → c'.count ≤ c.count

theorem max_quarters_is_twelve : 
  max_quarters ∧ ∃ (c : CoinCount), c.count = 12 ∧ total_value c = 480 :=
sorry


end max_quarters_is_twelve_l2512_251293


namespace largest_decimal_l2512_251297

theorem largest_decimal (a b c d e : ℚ) 
  (ha : a = 0.803) 
  (hb : b = 0.809) 
  (hc : c = 0.8039) 
  (hd : d = 0.8091) 
  (he : e = 0.8029) : 
  c = max a (max b (max c (max d e))) :=
by sorry

end largest_decimal_l2512_251297


namespace line_equation_proof_l2512_251227

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 1)

-- Define the y-intercept
def y_intercept : ℝ := -5

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := 6 * x - y - 5 = 0

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
    (x, y) = intersection_point →
    line1 x y ∧ line2 x y →
    target_line 0 y_intercept →
    target_line x y :=
by
  sorry

end line_equation_proof_l2512_251227


namespace sides_ratio_inscribed_circle_radius_l2512_251277

/-- A right-angled triangle with sides in arithmetic progression -/
structure ArithmeticRightTriangle where
  /-- The common difference of the arithmetic sequence -/
  d : ℝ
  /-- The common difference is positive -/
  d_pos : d > 0
  /-- The shortest side of the triangle -/
  shortest_side : ℝ
  /-- The shortest side is equal to 3d -/
  shortest_side_eq : shortest_side = 3 * d
  /-- The middle side of the triangle -/
  middle_side : ℝ
  /-- The middle side is equal to 4d -/
  middle_side_eq : middle_side = 4 * d
  /-- The longest side of the triangle (hypotenuse) -/
  longest_side : ℝ
  /-- The longest side is equal to 5d -/
  longest_side_eq : longest_side = 5 * d
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : shortest_side^2 + middle_side^2 = longest_side^2

/-- The ratio of sides in an ArithmeticRightTriangle is 3:4:5 -/
theorem sides_ratio (t : ArithmeticRightTriangle) :
  t.shortest_side / t.d = 3 ∧ t.middle_side / t.d = 4 ∧ t.longest_side / t.d = 5 := by
  sorry

/-- The radius of the inscribed circle is equal to the common difference -/
theorem inscribed_circle_radius (t : ArithmeticRightTriangle) :
  let s := (t.shortest_side + t.middle_side + t.longest_side) / 2
  let area := Real.sqrt (s * (s - t.shortest_side) * (s - t.middle_side) * (s - t.longest_side))
  area / s = t.d := by
  sorry

end sides_ratio_inscribed_circle_radius_l2512_251277


namespace total_food_eaten_l2512_251230

/-- The amount of food Ella eats in one day, in pounds. -/
def ellaFoodPerDay : ℝ := 20

/-- The ratio of food Ella's dog eats compared to Ella. -/
def dogFoodRatio : ℝ := 4

/-- The number of days considered. -/
def numDays : ℝ := 10

/-- Theorem stating the total amount of food eaten by Ella and her dog in the given period. -/
theorem total_food_eaten : 
  (ellaFoodPerDay * numDays) + (ellaFoodPerDay * dogFoodRatio * numDays) = 1000 := by
  sorry

end total_food_eaten_l2512_251230


namespace arithmetic_sequence_product_l2512_251287

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 5 * b 6 = 35 →
  b 4 * b 7 = 27 :=
by
  sorry

end arithmetic_sequence_product_l2512_251287


namespace solve_equation_l2512_251252

theorem solve_equation : ∃ x : ℚ, (3 * x) / 7 = 6 ∧ x = 14 := by sorry

end solve_equation_l2512_251252


namespace expected_adjacent_black_pairs_l2512_251294

theorem expected_adjacent_black_pairs (total_cards : ℕ) (black_cards : ℕ) (red_cards : ℕ)
  (h1 : total_cards = 60)
  (h2 : black_cards = 36)
  (h3 : red_cards = 24)
  (h4 : total_cards = black_cards + red_cards) :
  (black_cards : ℚ) * (black_cards - 1 : ℚ) / (total_cards - 1 : ℚ) = 1260 / 59 := by
sorry

end expected_adjacent_black_pairs_l2512_251294


namespace odd_function_period_range_l2512_251220

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, HasPeriod f q → q > 0 → p ≤ q

theorem odd_function_period_range (f : ℝ → ℝ) (m : ℝ) :
  IsOdd f →
  SmallestPositivePeriod f 3 →
  f 2015 > 1 →
  f 1 = (2 * m + 3) / (m - 1) →
  -2/3 < m ∧ m < 1 := by
  sorry

end odd_function_period_range_l2512_251220


namespace inequality_implication_l2512_251201

theorem inequality_implication (a b : ℝ) : a < b → -3 * a > -3 * b := by
  sorry

end inequality_implication_l2512_251201


namespace even_function_k_value_l2512_251208

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = kx^2 + (k - 1)x + 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 3

theorem even_function_k_value :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end even_function_k_value_l2512_251208


namespace additional_time_is_twelve_minutes_l2512_251209

/-- Represents the time (in hours) it takes for a worker to complete the job alone. -/
def completion_time (worker : ℕ) : ℚ :=
  match worker with
  | 1 => 4    -- P's completion time
  | 2 => 15   -- Q's completion time
  | _ => 0    -- Invalid worker

/-- Calculates the portion of the job completed by both workers in 3 hours. -/
def portion_completed : ℚ :=
  3 * ((1 / completion_time 1) + (1 / completion_time 2))

/-- Calculates the remaining portion of the job after 3 hours of joint work. -/
def remaining_portion : ℚ :=
  1 - portion_completed

/-- Calculates the additional time (in hours) needed for P to complete the remaining portion. -/
def additional_time : ℚ :=
  remaining_portion * completion_time 1

/-- The main theorem stating that the additional time for P to finish the job is 12 minutes. -/
theorem additional_time_is_twelve_minutes : additional_time * 60 = 12 := by
  sorry

end additional_time_is_twelve_minutes_l2512_251209


namespace f_symmetric_property_l2512_251206

/-- Given a function f(x) = ax^4 + bx^2 + 2x - 8 where a and b are real constants,
    if f(-1) = 10, then f(1) = -26 -/
theorem f_symmetric_property (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^4 + b * x^2 + 2 * x - 8
  f (-1) = 10 → f 1 = -26 := by
  sorry

end f_symmetric_property_l2512_251206


namespace badge_exchange_l2512_251261

theorem badge_exchange (V T : ℕ) : 
  V = T + 5 ∧ 
  (V - V * 24 / 100 + T * 20 / 100 : ℚ) = (T - T * 20 / 100 + V * 24 / 100 : ℚ) - 1 →
  V = 50 ∧ T = 45 :=
by sorry

end badge_exchange_l2512_251261


namespace burglar_sentence_l2512_251240

def painting_values : List ℝ := [9385, 12470, 7655, 8120, 13880]
def base_sentence_rate : ℝ := 3000
def assault_sentence : ℝ := 1.5
def resisting_arrest_sentence : ℝ := 2
def prior_offense_penalty : ℝ := 0.25

def calculate_total_sentence (values : List ℝ) (rate : ℝ) (assault : ℝ) (resisting : ℝ) (penalty : ℝ) : ℕ :=
  sorry

theorem burglar_sentence :
  calculate_total_sentence painting_values base_sentence_rate assault_sentence resisting_arrest_sentence prior_offense_penalty = 26 :=
sorry

end burglar_sentence_l2512_251240


namespace floor_equation_solution_l2512_251296

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 2 ≤ x ∧ x < 7/3 := by sorry

end floor_equation_solution_l2512_251296


namespace christmas_gifts_left_l2512_251204

/-- The number of gifts left under the Christmas tree -/
def gifts_left (initial : ℕ) (sent : ℕ) : ℕ := initial - sent

/-- Theorem stating that given 77 initial gifts and 66 sent gifts, 11 gifts are left -/
theorem christmas_gifts_left : gifts_left 77 66 = 11 := by
  sorry

end christmas_gifts_left_l2512_251204


namespace million_place_seven_digits_l2512_251280

/-- A place value in a number system. -/
inductive PlaceValue
  | Units
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands
  | Millions

/-- The number of digits in a place value. -/
def PlaceValue.digits : PlaceValue → Nat
  | Units => 1
  | Tens => 2
  | Hundreds => 3
  | Thousands => 4
  | TenThousands => 5
  | HundredThousands => 6
  | Millions => 7

/-- A number with its highest place being the million place has 7 digits. -/
theorem million_place_seven_digits :
  PlaceValue.digits PlaceValue.Millions = 7 := by
  sorry

end million_place_seven_digits_l2512_251280


namespace al_sandwich_options_l2512_251266

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 4

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 6

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 5

/-- Represents whether ham is available. -/
def has_ham : Prop := True

/-- Represents whether chicken is available. -/
def has_chicken : Prop := True

/-- Represents whether cheddar cheese is available. -/
def has_cheddar : Prop := True

/-- Represents whether white bread is available. -/
def has_white_bread : Prop := True

/-- Represents the number of sandwiches with ham and cheddar cheese combination. -/
def ham_cheddar_combos : ℕ := num_bread

/-- Represents the number of sandwiches with white bread and chicken combination. -/
def white_chicken_combos : ℕ := num_cheese

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem al_sandwich_options : 
  num_bread * num_meat * num_cheese - ham_cheddar_combos - white_chicken_combos = 111 := by
  sorry

end al_sandwich_options_l2512_251266


namespace missing_number_proof_l2512_251298

theorem missing_number_proof (numbers : List ℕ) (missing : ℕ) : 
  numbers = [744, 745, 747, 748, 749, 752, 752, 753, 755] →
  (numbers.sum + missing) / 10 = 750 →
  missing = 805 := by
sorry

end missing_number_proof_l2512_251298


namespace license_plate_palindrome_probability_l2512_251247

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def plate_length : ℕ := 4

def prob_digit_palindrome : ℚ := (digit_count ^ 2) / (digit_count ^ plate_length)
def prob_letter_palindrome : ℚ := (letter_count ^ 2) / (letter_count ^ plate_length)

theorem license_plate_palindrome_probability :
  let prob_at_least_one_palindrome := prob_digit_palindrome + prob_letter_palindrome - 
    (prob_digit_palindrome * prob_letter_palindrome)
  prob_at_least_one_palindrome = 97 / 8450 := by
  sorry

end license_plate_palindrome_probability_l2512_251247


namespace x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l2512_251284

theorem x_lt_1_necessary_not_sufficient_for_ln_x_lt_0 :
  (∀ x : ℝ, Real.log x < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ Real.log x ≥ 0) :=
by sorry

end x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l2512_251284


namespace shape_selections_count_l2512_251210

/-- Represents a regular hexagonal grid --/
structure HexagonalGrid :=
  (size : ℕ)

/-- Represents a shape that can be selected from the grid --/
structure Shape :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the number of ways to select a shape from a hexagonal grid --/
def selectionsCount (grid : HexagonalGrid) (shape : Shape) : ℕ :=
  sorry

/-- The number of distinct rotations for a shape in a hexagonal grid --/
def rotationsCount : ℕ := 3

/-- Theorem stating that there are 72 ways to select the given shape from the hexagonal grid --/
theorem shape_selections_count :
  ∀ (grid : HexagonalGrid) (shape : Shape),
  grid.size = 5 →  -- Assuming the grid size is 5 based on the problem description
  shape.width = 2 →  -- Assuming the shape width is 2 based on diagram b
  shape.height = 2 →  -- Assuming the shape height is 2 based on diagram b
  selectionsCount grid shape * rotationsCount = 72 :=
sorry

end shape_selections_count_l2512_251210


namespace power_of_i_product_l2512_251282

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_product : (i^15) * (i^135) = -1 := by
  sorry

end power_of_i_product_l2512_251282


namespace polygon_sides_l2512_251246

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : 
  sum_interior_angles = 1080 → n = 8 :=
by
  sorry

end polygon_sides_l2512_251246


namespace order_of_logarithmic_fractions_l2512_251216

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := (Real.log 3) / 3
  let c : ℝ := (Real.log 5) / 5
  c < a ∧ a < b := by sorry

end order_of_logarithmic_fractions_l2512_251216


namespace ratio_problem_l2512_251271

theorem ratio_problem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ratio : (x + y) / (x - y) = 4 / 3) : x / y = 7 := by
  sorry

end ratio_problem_l2512_251271


namespace chang_e_2_orbit_period_l2512_251257

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 + 1 / (α n + 1 / b α n)

theorem chang_e_2_orbit_period (α : ℕ → ℕ) :
  b α 4 < b α 7 := by
  sorry

end chang_e_2_orbit_period_l2512_251257


namespace evaluate_expression_l2512_251251

theorem evaluate_expression : 
  Real.sqrt ((5 - 3 * Real.sqrt 5) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 5) ^ 2) = -10 := by
  sorry

end evaluate_expression_l2512_251251


namespace distribution_of_four_men_five_women_l2512_251215

/-- The number of ways to distribute men and women into groups -/
def group_distribution (men women : ℕ) : ℕ :=
  let group_of_two := men.choose 1 * women.choose 1
  let group_of_three_1 := (men - 1).choose 2 * (women - 1).choose 1
  let group_of_three_2 := 1 * (women - 2).choose 2
  (group_of_two * group_of_three_1 * group_of_three_2) / 2

/-- Theorem stating the number of ways to distribute 4 men and 5 women -/
theorem distribution_of_four_men_five_women :
  group_distribution 4 5 = 360 := by
  sorry

#eval group_distribution 4 5

end distribution_of_four_men_five_women_l2512_251215


namespace olivia_wednesday_hours_l2512_251248

/-- Calculates the number of hours Olivia worked on Wednesday -/
def wednesday_hours (hourly_rate : ℕ) (monday_hours : ℕ) (friday_hours : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - hourly_rate * (monday_hours + friday_hours)) / hourly_rate

/-- Proves that Olivia worked 3 hours on Wednesday given the conditions -/
theorem olivia_wednesday_hours :
  wednesday_hours 9 4 6 117 = 3 := by
sorry

end olivia_wednesday_hours_l2512_251248


namespace focus_to_asymptote_distance_l2512_251285

/-- Given a hyperbola with equation x²/(3a) - y²/a = 1 where a > 0,
    the distance from a focus to an asymptote is √a -/
theorem focus_to_asymptote_distance (a : ℝ) (ha : a > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / (3*a) - y^2 / a = 1}
  let focus : ℝ × ℝ := (2 * Real.sqrt a, 0)
  let asymptote := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x}
  ∃ (p : ℝ × ℝ), p ∈ asymptote ∧ 
    Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2) = Real.sqrt a :=
sorry

end focus_to_asymptote_distance_l2512_251285


namespace linear_system_solution_l2512_251290

theorem linear_system_solution (u v : ℚ) 
  (eq1 : 6 * u - 7 * v = 32)
  (eq2 : 3 * u + 5 * v = 1) : 
  2 * u + 3 * v = 64 / 51 := by
  sorry

end linear_system_solution_l2512_251290


namespace distinct_cube_digits_mod_seven_l2512_251222

theorem distinct_cube_digits_mod_seven :
  ∃! s : Finset ℕ, 
    (∀ n : ℕ, (n^3 % 7) ∈ s) ∧ 
    (∀ m ∈ s, ∃ n : ℕ, n^3 % 7 = m) ∧
    s.card = 3 := by
  sorry

end distinct_cube_digits_mod_seven_l2512_251222


namespace angle_terminal_side_point_angle_terminal_side_point_with_sin_l2512_251273

-- Part 1
theorem angle_terminal_side_point (α : Real) :
  ∃ (P : ℝ × ℝ), P.1 = 4 ∧ P.2 = -3 →
  2 * Real.sin α + Real.cos α = -2/5 := by sorry

-- Part 2
theorem angle_terminal_side_point_with_sin (α : Real) (m : Real) :
  m ≠ 0 →
  ∃ (P : ℝ × ℝ), P.1 = -Real.sqrt 3 ∧ P.2 = m →
  Real.sin α = (Real.sqrt 2 * m) / 4 →
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  (m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
  (m < 0 → Real.tan α = Real.sqrt 15 / 3) := by sorry

end angle_terminal_side_point_angle_terminal_side_point_with_sin_l2512_251273


namespace congruent_triangles_on_skew_lines_l2512_251223

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (g l : Line3D) : Prop := sorry

/-- A point lies on a line in 3D space. -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- Two triangles in 3D space are congruent. -/
def triangles_congruent (t1 t2 : Triangle3D) : Prop := sorry

/-- The number of congruent triangles that can be constructed on two skew lines. -/
def num_congruent_triangles_on_skew_lines (g l : Line3D) (abc : Triangle3D) : ℕ := sorry

/-- Theorem: Given two skew lines and a triangle, there exist exactly 16 congruent triangles
    with vertices on the given lines. -/
theorem congruent_triangles_on_skew_lines (g l : Line3D) (abc : Triangle3D) :
  are_skew g l →
  num_congruent_triangles_on_skew_lines g l abc = 16 :=
by sorry

end congruent_triangles_on_skew_lines_l2512_251223


namespace isosceles_triangle_base_angle_l2512_251260

structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)
  sumIs180 : angle1 + angle2 + angle3 = 180

theorem isosceles_triangle_base_angle 
  (triangle : IsoscelesTriangle) 
  (has80DegreeAngle : triangle.angle1 = 80 ∨ triangle.angle2 = 80 ∨ triangle.angle3 = 80) :
  (∃ baseAngle : ℝ, (baseAngle = 80 ∨ baseAngle = 50) ∧ 
   ((triangle.angle1 = baseAngle ∧ triangle.angle2 = baseAngle) ∨
    (triangle.angle1 = baseAngle ∧ triangle.angle3 = baseAngle) ∨
    (triangle.angle2 = baseAngle ∧ triangle.angle3 = baseAngle))) :=
by sorry

end isosceles_triangle_base_angle_l2512_251260


namespace min_value_theorem_l2512_251212

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 4) :
  ((x + 1) * (2*y + 1)) / (x * y) ≥ 9/2 :=
sorry

end min_value_theorem_l2512_251212


namespace two_digit_number_digit_difference_l2512_251244

theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 36 → x - y = 4 := by
  sorry

end two_digit_number_digit_difference_l2512_251244


namespace set_membership_implies_value_l2512_251211

theorem set_membership_implies_value (m : ℤ) : 
  3 ∈ ({1, m+2} : Set ℤ) → m = 1 := by
  sorry

end set_membership_implies_value_l2512_251211


namespace sum_of_factors_l2512_251205

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120 →
  a + b + c + d + e = 13 := by
sorry

end sum_of_factors_l2512_251205


namespace smallest_circle_radius_l2512_251256

/-- The radius of the smallest circle containing a triangle with sides 7, 9, and 12 -/
theorem smallest_circle_radius (a b c : ℝ) (ha : a = 7) (hb : b = 9) (hc : c = 12) :
  let R := max a (max b c) / 2
  R = 6 := by sorry

end smallest_circle_radius_l2512_251256


namespace people_after_yoongi_l2512_251218

theorem people_after_yoongi (total : ℕ) (before : ℕ) (h1 : total = 20) (h2 : before = 11) :
  total - (before + 1) = 8 := by
  sorry

end people_after_yoongi_l2512_251218


namespace same_units_digit_count_l2512_251213

def old_page_numbers := Finset.range 60

theorem same_units_digit_count :
  (old_page_numbers.filter (λ x => x % 10 = (61 - x) % 10)).card = 6 := by
  sorry

end same_units_digit_count_l2512_251213


namespace johns_computer_purchase_cost_l2512_251217

theorem johns_computer_purchase_cost
  (computer_cost : ℝ)
  (peripherals_cost_ratio : ℝ)
  (original_video_card_cost : ℝ)
  (upgraded_video_card_cost_ratio : ℝ)
  (video_card_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : peripherals_cost_ratio = 1 / 4)
  (h3 : original_video_card_cost = 300)
  (h4 : upgraded_video_card_cost_ratio = 2.5)
  (h5 : video_card_discount_rate = 0.12)
  (h6 : sales_tax_rate = 0.05) :
  let peripherals_cost := computer_cost * peripherals_cost_ratio
  let upgraded_video_card_cost := original_video_card_cost * upgraded_video_card_cost_ratio
  let video_card_discount := upgraded_video_card_cost * video_card_discount_rate
  let final_video_card_cost := upgraded_video_card_cost - video_card_discount
  let sales_tax := peripherals_cost * sales_tax_rate
  let total_cost := computer_cost + peripherals_cost + final_video_card_cost + sales_tax
  total_cost = 2553.75 :=
by sorry

end johns_computer_purchase_cost_l2512_251217


namespace intersection_of_M_and_N_l2512_251233

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l2512_251233


namespace yolanda_bike_speed_yolanda_speed_equals_husband_speed_l2512_251207

/-- Yolanda's bike ride problem -/
theorem yolanda_bike_speed (husband_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  husband_speed > 0 ∧ head_start > 0 ∧ catch_up_time > 0 →
  ∃ (bike_speed : ℝ),
    bike_speed > 0 ∧
    bike_speed * (head_start + catch_up_time) = husband_speed * catch_up_time :=
by
  sorry

/-- Yolanda's bike speed is equal to her husband's car speed -/
theorem yolanda_speed_equals_husband_speed :
  ∃ (bike_speed : ℝ),
    bike_speed > 0 ∧
    bike_speed = 40 ∧
    bike_speed * (15/60 + 15/60) = 40 * (15/60) :=
by
  sorry

end yolanda_bike_speed_yolanda_speed_equals_husband_speed_l2512_251207


namespace professor_seating_arrangements_l2512_251268

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 9

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the condition that professors cannot sit in the first or last chair -/
def available_chairs : ℕ := total_chairs - 2

/-- Represents the effective number of chair choices after accounting for spacing -/
def effective_choices : ℕ := available_chairs - (num_professors - 1)

/-- The number of ways to choose professor positions -/
def choose_positions : ℕ := Nat.choose effective_choices num_professors

/-- The number of ways to arrange professors in the chosen positions -/
def arrange_professors : ℕ := Nat.factorial num_professors

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  choose_positions * arrange_professors = 60 := by sorry

end professor_seating_arrangements_l2512_251268


namespace simplify_expression_l2512_251265

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = -24 / 5 := by
  sorry

end simplify_expression_l2512_251265


namespace divisibility_property_l2512_251242

theorem divisibility_property (n : ℕ) : 
  ∃ (a b : ℕ), (a * n + 1)^6 + b ≡ 0 [MOD (n^2 + n + 1)] :=
by
  use 2, 27
  sorry

end divisibility_property_l2512_251242


namespace nina_calculation_l2512_251226

theorem nina_calculation (y : ℚ) : (y + 25) * 5 = 200 → (y - 25) / 5 = -2 := by
  sorry

end nina_calculation_l2512_251226


namespace air_conditioner_problem_l2512_251229

/-- Represents the selling prices and quantities of air conditioners --/
structure AirConditioner where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Represents the cost prices of air conditioners --/
structure CostPrices where
  cost_A : ℝ
  cost_B : ℝ

/-- The theorem statement for the air conditioner problem --/
theorem air_conditioner_problem 
  (sale1 : AirConditioner)
  (sale2 : AirConditioner)
  (costs : CostPrices)
  (h1 : sale1.quantity_A = 3 ∧ sale1.quantity_B = 5)
  (h2 : sale2.quantity_A = 4 ∧ sale2.quantity_B = 10)
  (h3 : sale1.price_A * sale1.quantity_A + sale1.price_B * sale1.quantity_B = 23500)
  (h4 : sale2.price_A * sale2.quantity_A + sale2.price_B * sale2.quantity_B = 42000)
  (h5 : costs.cost_A = 1800 ∧ costs.cost_B = 2400)
  (h6 : sale1.price_A = sale2.price_A ∧ sale1.price_B = sale2.price_B) :
  sale1.price_A = 2500 ∧ 
  sale1.price_B = 3200 ∧ 
  (∃ m : ℕ, 
    m ≥ 30 ∧ 
    (sale1.price_A - costs.cost_A) * (50 - m) + (sale1.price_B - costs.cost_B) * m ≥ 38000 ∧
    ∀ n : ℕ, n < 30 → (sale1.price_A - costs.cost_A) * (50 - n) + (sale1.price_B - costs.cost_B) * n < 38000) := by
  sorry

end air_conditioner_problem_l2512_251229


namespace base_7_to_decimal_l2512_251214

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem base_7_to_decimal :
  to_decimal [6, 5, 7] 7 = 384 := by
  sorry

end base_7_to_decimal_l2512_251214


namespace functional_equation_solution_l2512_251275

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → (x + y) ∈ (Set.Ioo (-1) 1) →
    f (x + y) = (f x + f y) / (1 - f x * f y)

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, ContinuousOn f (Set.Ioo (-1) 1) →
  FunctionalEquation f →
  ∃ a : ℝ, |a| ≤ π/2 ∧ ∀ x ∈ (Set.Ioo (-1) 1), f x = Real.tan (a * x) := by
  sorry

end functional_equation_solution_l2512_251275


namespace sum_of_a_and_b_is_negative_five_l2512_251267

-- Define the sets P and Q
def P : Set ℝ := {y | y^2 - y - 2 > 0}
def Q : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_a_and_b_is_negative_five 
  (h1 : P ∪ Q = Set.univ)
  (h2 : P ∩ Q = Set.Ioc 2 3)
  : ∃ (a b : ℝ), Q = {x | x^2 + a*x + b ≤ 0} ∧ a + b = -5 :=
sorry

end sum_of_a_and_b_is_negative_five_l2512_251267


namespace smallest_class_size_l2512_251232

theorem smallest_class_size (n : ℕ) : 
  (∃ (m : ℕ), 4 * n + (n + 1) = m ∧ m > 40) → 
  (∀ (k : ℕ), k < n → ¬(∃ (m : ℕ), 4 * k + (k + 1) = m ∧ m > 40)) → 
  4 * n + (n + 1) = 41 :=
sorry

end smallest_class_size_l2512_251232


namespace arithmetic_sequence_length_l2512_251200

/-- Given an arithmetic sequence with first term 2, last term 2014, and common difference 3,
    prove that it has 671 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 2 →                        -- First term is 2
    (∀ n, a (n + 1) = a n + 3) →     -- Common difference is 3
    (∃ k, a k = 2014) →              -- Last term is 2014
    (∃ k, a k = 2014 ∧ k + 1 = 671)  -- The sequence has 671 terms
    := by sorry

end arithmetic_sequence_length_l2512_251200
