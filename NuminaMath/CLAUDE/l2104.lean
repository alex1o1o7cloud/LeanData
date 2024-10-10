import Mathlib

namespace inlet_pipe_rate_l2104_210446

/-- Prove that the inlet pipe rate is 3 cubic inches/min given the tank and pipe conditions -/
theorem inlet_pipe_rate (tank_volume : ℝ) (outlet_rate1 outlet_rate2 : ℝ) (empty_time : ℝ) :
  tank_volume = 51840 ∧ 
  outlet_rate1 = 9 ∧ 
  outlet_rate2 = 6 ∧ 
  empty_time = 4320 →
  ∃ inlet_rate : ℝ, 
    inlet_rate = 3 ∧ 
    (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time = tank_volume :=
by
  sorry

end inlet_pipe_rate_l2104_210446


namespace prob_king_queen_heart_l2104_210491

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Number of Hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of King of Hearts in a standard deck -/
def NumKingOfHearts : ℕ := 1

/-- Probability of drawing a King, then a Queen, then a Heart from a standard 52-card deck -/
theorem prob_king_queen_heart : 
  (NumKings * (NumQueens - 1) * NumHearts + 
   NumKingOfHearts * NumQueens * (NumHearts - 1) + 
   NumKingOfHearts * (NumQueens - 1) * (NumHearts - 1)) / 
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 67 / 44200 := by
  sorry

end prob_king_queen_heart_l2104_210491


namespace radio_selling_price_l2104_210430

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def selling_price (cost_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  cost_price * (1 - loss_percentage / 100)

/-- Theorem: The selling price of a radio with a cost price of 1500 and a loss percentage of 17 is 1245. -/
theorem radio_selling_price :
  selling_price 1500 17 = 1245 := by
  sorry

end radio_selling_price_l2104_210430


namespace total_repair_cost_is_50_95_l2104_210492

def tire_repair_cost (num_tires : ℕ) (cost_per_tire : ℚ) (sales_tax : ℚ) 
                     (discount_rate : ℚ) (discount_valid : Bool) (city_fee : ℚ) : ℚ :=
  let base_cost := num_tires * cost_per_tire
  let tax_cost := num_tires * sales_tax
  let fee_cost := num_tires * city_fee
  let discount := if discount_valid then discount_rate * base_cost else 0
  base_cost + tax_cost + fee_cost - discount

theorem total_repair_cost_is_50_95 :
  let car_a_cost := tire_repair_cost 3 7 0.5 0.05 true 2.5
  let car_b_cost := tire_repair_cost 2 8.5 0 0.1 false 2.5
  car_a_cost + car_b_cost = 50.95 := by
sorry

#eval tire_repair_cost 3 7 0.5 0.05 true 2.5 + tire_repair_cost 2 8.5 0 0.1 false 2.5

end total_repair_cost_is_50_95_l2104_210492


namespace perpendicular_lines_from_perpendicular_planes_l2104_210464

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (plane_perp : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (line_perp : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : line_perp_plane m α)
  (h2 : line_perp_plane n β)
  (h3 : plane_perp α β) :
  line_perp m n :=
sorry

end perpendicular_lines_from_perpendicular_planes_l2104_210464


namespace saree_price_calculation_l2104_210450

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.20) * (1 - 0.05) = 133) → P = 175 := by
  sorry

end saree_price_calculation_l2104_210450


namespace school_distance_l2104_210473

theorem school_distance (speed_to_school : ℝ) (speed_from_school : ℝ) (total_time : ℝ) 
  (h1 : speed_to_school = 3)
  (h2 : speed_from_school = 2)
  (h3 : total_time = 5) :
  ∃ (distance : ℝ), distance = 6 ∧ 
    (distance / speed_to_school + distance / speed_from_school = total_time) :=
by
  sorry

end school_distance_l2104_210473


namespace david_average_speed_l2104_210403

/-- Calculates the average speed given distance and time -/
def average_speed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

/-- Converts hours and minutes to hours -/
def hours_and_minutes_to_hours (hours : ℕ) (minutes : ℕ) : ℚ :=
  hours + (minutes : ℚ) / 60

theorem david_average_speed :
  let distance : ℚ := 49/3  -- 16 1/3 miles as an improper fraction
  let time : ℚ := hours_and_minutes_to_hours 2 20
  average_speed distance time = 7 := by sorry

end david_average_speed_l2104_210403


namespace tan_sum_simplification_l2104_210487

theorem tan_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = (Real.sqrt 3 + 1) / (Real.sqrt 3 * Real.cos (40 * π / 180)) := by
  sorry

end tan_sum_simplification_l2104_210487


namespace continuous_fraction_solution_l2104_210412

theorem continuous_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (6 + 2 * Real.sqrt 39) / 4 := by
  sorry

end continuous_fraction_solution_l2104_210412


namespace job_choice_diploma_percentage_l2104_210426

theorem job_choice_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_with_job : ℝ := 12
  let with_job_choice : ℝ := 40
  let with_diploma : ℝ := 43
  let without_job_choice : ℝ := total_population - with_job_choice
  let with_diploma_and_job : ℝ := with_job_choice - no_diploma_with_job
  let with_diploma_without_job : ℝ := with_diploma - with_diploma_and_job
  (with_diploma_without_job / without_job_choice) * 100 = 25 := by
  sorry

end job_choice_diploma_percentage_l2104_210426


namespace ellipse_properties_l2104_210400

/-- Given an ellipse with the following properties:
  * Equation: x²/a² + y²/b² = 1, where a > b > 0
  * Vertices: A(0,b) and C(0,-b)
  * Foci: F₁(-c,0) and F₂(c,0), where c > 0
  * A line through point E(3c,0) intersects the ellipse at another point B
  * F₁A ∥ F₂B
-/
theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let e := c / a -- eccentricity
  let m := (5/3) * c
  let n := (2*Real.sqrt 2/3) * c
  ∃ (x y : ℝ),
    -- Point B on the ellipse
    x^2/a^2 + y^2/b^2 = 1 ∧
    -- B is on the line through E(3c,0)
    ∃ (t : ℝ), x = 3*c*(1-t) ∧ y = 3*c*t ∧
    -- F₁A ∥ F₂B
    (b / (-c)) = (y - 0) / (x - c) ∧
    -- Eccentricity is √3/3
    e = Real.sqrt 3 / 3 ∧
    -- Point H(m,n) is on F₂B
    n / (m - c) = y / (x - c) ∧
    -- H is on the circumcircle of AF₁C
    (m - c/2)^2 + n^2 = (3*c/2)^2 ∧
    -- Ratio n/m
    n / m = 2 * Real.sqrt 2 / 5 := by
  sorry

end ellipse_properties_l2104_210400


namespace selling_price_ratio_l2104_210476

theorem selling_price_ratio (CP : ℝ) (SP1 SP2 : ℝ) 
  (h1 : SP1 = CP + 0.5 * CP) 
  (h2 : SP2 = CP + 3 * CP) : 
  SP2 / SP1 = 8 / 3 := by
sorry

end selling_price_ratio_l2104_210476


namespace aunt_gemma_dog_food_l2104_210495

/-- The number of sacks of dog food Aunt Gemma bought -/
def num_sacks : ℕ := 2

/-- The number of dogs Aunt Gemma has -/
def num_dogs : ℕ := 4

/-- The number of times Aunt Gemma feeds her dogs per day -/
def feeds_per_day : ℕ := 2

/-- The amount of food each dog consumes per meal in grams -/
def food_per_meal : ℕ := 250

/-- The weight of each sack of dog food in kilograms -/
def sack_weight : ℕ := 50

/-- The number of days the dog food will last -/
def days_lasting : ℕ := 50

theorem aunt_gemma_dog_food :
  num_sacks = (num_dogs * feeds_per_day * food_per_meal * days_lasting) / (sack_weight * 1000) := by
  sorry

end aunt_gemma_dog_food_l2104_210495


namespace profit_rate_change_is_three_percent_l2104_210414

/-- Represents the change in profit rate that causes A's income to increase by 300 --/
def profit_rate_change (a_share : ℚ) (a_capital : ℕ) (income_increase : ℕ) : ℚ :=
  (income_increase : ℚ) / a_capital / a_share * 100

/-- Theorem stating the change in profit rate given the problem conditions --/
theorem profit_rate_change_is_three_percent :
  profit_rate_change (2/3) 15000 300 = 3 := by
  sorry

end profit_rate_change_is_three_percent_l2104_210414


namespace widget_difference_formula_l2104_210421

/-- Represents the widget production difference between Monday and Tuesday -/
def widget_difference (t : ℝ) : ℝ :=
  let w : ℝ := 3 * t - 1
  let monday_production : ℝ := w * t
  let tuesday_production : ℝ := (w + 6) * (t - 3)
  monday_production - tuesday_production

/-- Theorem stating the widget production difference -/
theorem widget_difference_formula (t : ℝ) :
  widget_difference t = 3 * t + 15 := by
  sorry

#check widget_difference_formula

end widget_difference_formula_l2104_210421


namespace large_long_furred_brown_dogs_l2104_210470

/-- Represents the characteristics of dogs in a kennel -/
structure DogKennel where
  total : ℕ
  longFurred : ℕ
  brown : ℕ
  neitherLongFurredNorBrown : ℕ
  large : ℕ
  small : ℕ
  smallAndBrown : ℕ
  onlyLargeAndLongFurred : ℕ

/-- Theorem stating the number of large, long-furred, brown dogs -/
theorem large_long_furred_brown_dogs (k : DogKennel)
  (h1 : k.total = 60)
  (h2 : k.longFurred = 35)
  (h3 : k.brown = 25)
  (h4 : k.neitherLongFurredNorBrown = 10)
  (h5 : k.large = 30)
  (h6 : k.small = 30)
  (h7 : k.smallAndBrown = 14)
  (h8 : k.onlyLargeAndLongFurred = 7) :
  ∃ n : ℕ, n = 6 ∧ n = k.large - k.onlyLargeAndLongFurred - (k.brown - k.smallAndBrown) :=
by sorry


end large_long_furred_brown_dogs_l2104_210470


namespace triangle_angle_B_l2104_210435

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) : 
  A = π/4 → a = 6 → b = 3 * Real.sqrt 2 → 
  0 < A ∧ A < π → 0 < B ∧ B < π → 0 < C ∧ C < π →
  a * Real.sin B = b * Real.sin A → 
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  B = π/6 := by sorry

end triangle_angle_B_l2104_210435


namespace average_height_theorem_l2104_210441

def tree_heights (h₁ h₂ h₃ h₄ h₅ : ℝ) : Prop :=
  h₂ = 15 ∧
  (h₂ = h₁ + 5 ∨ h₂ = h₁ - 3) ∧
  (h₃ = h₂ + 5 ∨ h₃ = h₂ - 3) ∧
  (h₄ = h₃ + 5 ∨ h₄ = h₃ - 3) ∧
  (h₅ = h₄ + 5 ∨ h₅ = h₄ - 3)

theorem average_height_theorem (h₁ h₂ h₃ h₄ h₅ : ℝ) :
  tree_heights h₁ h₂ h₃ h₄ h₅ →
  ∃ (k : ℤ), (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = k + 0.4 →
  (h₁ + h₂ + h₃ + h₄ + h₅) / 5 = 20.4 :=
by sorry

end average_height_theorem_l2104_210441


namespace quadratic_root_bounds_l2104_210493

theorem quadratic_root_bounds (a b : ℝ) (α β : ℝ) : 
  (α^2 + a*α + b = 0) → 
  (β^2 + a*β + b = 0) → 
  (∀ x, x^2 + a*x + b = 0 → x = α ∨ x = β) →
  (|α| < 2 ∧ |β| < 2 ↔ 2*|a| < 4 + b ∧ |b| < 4) := by
sorry

end quadratic_root_bounds_l2104_210493


namespace sum_real_imag_parts_of_complex_l2104_210417

theorem sum_real_imag_parts_of_complex (z : ℂ) : z = (1 + 2*I) / (1 - 2*I) → (z.re + z.im = 1/5) := by
  sorry

end sum_real_imag_parts_of_complex_l2104_210417


namespace integer_square_root_of_seven_minus_x_l2104_210419

theorem integer_square_root_of_seven_minus_x (x : ℕ+) :
  (∃ (n : ℤ), n^2 = 7 - x.val) → x.val = 3 ∨ x.val = 6 ∨ x.val = 7 := by
  sorry

end integer_square_root_of_seven_minus_x_l2104_210419


namespace substitution_method_correctness_l2104_210483

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 5
def equation2 (x y : ℝ) : Prop := y = 1 + x

-- Define the correct substitution
def correct_substitution (x : ℝ) : Prop := 2 * x - 1 - x = 5

-- Theorem statement
theorem substitution_method_correctness :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → correct_substitution x :=
by sorry

end substitution_method_correctness_l2104_210483


namespace sum_of_four_numbers_l2104_210467

theorem sum_of_four_numbers : 2367 + 3672 + 6723 + 7236 = 19998 := by
  sorry

end sum_of_four_numbers_l2104_210467


namespace bread_pieces_theorem_l2104_210463

/-- The number of pieces a slice of bread becomes when torn in half twice -/
def pieces_per_slice : ℕ := 4

/-- The number of slices of bread used -/
def num_slices : ℕ := 2

/-- The total number of bread pieces after tearing -/
def total_pieces : ℕ := num_slices * pieces_per_slice

theorem bread_pieces_theorem : total_pieces = 8 := by
  sorry

end bread_pieces_theorem_l2104_210463


namespace tan_fifteen_fraction_equals_sqrt_three_over_three_l2104_210409

theorem tan_fifteen_fraction_equals_sqrt_three_over_three :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end tan_fifteen_fraction_equals_sqrt_three_over_three_l2104_210409


namespace exactly_two_rotational_homotheties_l2104_210440

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rotational homothety with 90° rotation --/
structure RotationalHomothety where
  center : ℝ × ℝ
  scale : ℝ

/-- The number of rotational homotheties with 90° rotation that map one circle to another --/
def num_rotational_homotheties (S₁ S₂ : Circle) : ℕ :=
  sorry

/-- Two circles are non-concentric if their centers are different --/
def non_concentric (S₁ S₂ : Circle) : Prop :=
  S₁.center ≠ S₂.center

theorem exactly_two_rotational_homotheties (S₁ S₂ : Circle) 
  (h : non_concentric S₁ S₂) : num_rotational_homotheties S₁ S₂ = 2 := by
  sorry

end exactly_two_rotational_homotheties_l2104_210440


namespace mass_percentage_cl_l2104_210490

/-- Given a compound where the mass percentage of Cl is 92.11%,
    prove that the mass percentage of Cl in the compound is 92.11%. -/
theorem mass_percentage_cl (compound_mass_percentage : ℝ) 
  (h : compound_mass_percentage = 92.11) : 
  compound_mass_percentage = 92.11 := by
  sorry

end mass_percentage_cl_l2104_210490


namespace min_value_of_expression_min_value_achievable_l2104_210420

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (1 / (1 + a)) + (4 / (4 + b)) ≥ 9/8 := by sorry

theorem min_value_achievable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧
  (1 / (1 + a₀)) + (4 / (4 + b₀)) = 9/8 := by sorry

end min_value_of_expression_min_value_achievable_l2104_210420


namespace absolute_value_inequality_l2104_210431

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x - 2))| > 3 ↔ 2/3 < x ∧ x < 2 := by
  sorry

end absolute_value_inequality_l2104_210431


namespace reciprocal_difference_decreases_l2104_210457

theorem reciprocal_difference_decreases (n : ℕ) : 
  (1 : ℚ) / n - (1 : ℚ) / (n + 1) = 1 / (n * (n + 1)) ∧
  ∀ m : ℕ, m > n → (1 : ℚ) / m - (1 : ℚ) / (m + 1) < (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by sorry

end reciprocal_difference_decreases_l2104_210457


namespace pencil_eraser_notebook_cost_l2104_210488

theorem pencil_eraser_notebook_cost 
  (h1 : 20 * x + 3 * y + 2 * z = 32) 
  (h2 : 39 * x + 5 * y + 3 * z = 58) : 
  5 * x + 5 * y + 5 * z = 30 := by
  sorry

end pencil_eraser_notebook_cost_l2104_210488


namespace simplify_expression_l2104_210402

theorem simplify_expression (x : ℝ) : 
  2*x + 4*x^2 - 3 + (5 - 3*x - 9*x^2) + (x+1)^2 = -4*x^2 + x + 3 := by
  sorry

end simplify_expression_l2104_210402


namespace sum_of_xy_l2104_210472

theorem sum_of_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_bound : x < 30) (hy_bound : y < 30) 
  (h_eq : x + y + x * y = 119) : x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := by
  sorry

end sum_of_xy_l2104_210472


namespace cubic_equation_root_l2104_210432

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5 : ℝ) ^ 3 + a * (3 + Real.sqrt 5 : ℝ) ^ 2 + b * (3 + Real.sqrt 5 : ℝ) - 20 = 0 → 
  b = -26 := by
sorry

end cubic_equation_root_l2104_210432


namespace shirts_purchased_l2104_210413

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_cost : ℕ := 51
def num_jeans : ℕ := 2
def num_hats : ℕ := 4

theorem shirts_purchased : 
  ∃ (num_shirts : ℕ), 
    num_shirts * shirt_cost + num_jeans * jeans_cost + num_hats * hat_cost = total_cost ∧ 
    num_shirts = 3 := by
  sorry

end shirts_purchased_l2104_210413


namespace marathon_duration_in_minutes_l2104_210428

-- Define the duration of the marathon
def marathon_hours : ℕ := 15
def marathon_minutes : ℕ := 35

-- Theorem to prove
theorem marathon_duration_in_minutes :
  marathon_hours * 60 + marathon_minutes = 935 := by
  sorry

end marathon_duration_in_minutes_l2104_210428


namespace money_distribution_l2104_210444

theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 → A + C = 200 → B + C = 330 → C = 30 := by
  sorry

end money_distribution_l2104_210444


namespace different_color_probability_l2104_210405

/-- The probability of drawing two chips of different colors from a bag -/
theorem different_color_probability (total_chips : ℕ) (blue_chips : ℕ) (yellow_chips : ℕ) 
  (h1 : total_chips = blue_chips + yellow_chips)
  (h2 : blue_chips = 7)
  (h3 : yellow_chips = 2) :
  (blue_chips * yellow_chips + yellow_chips * blue_chips) / (total_chips * (total_chips - 1)) = 7 / 18 := by
  sorry

end different_color_probability_l2104_210405


namespace sparrow_swallow_weight_system_l2104_210479

theorem sparrow_swallow_weight_system :
  ∀ (x y : ℝ),
    (∃ (sparrow_count swallow_count : ℕ),
      sparrow_count = 5 ∧
      swallow_count = 6 ∧
      (4 * x + y = 5 * y + x) ∧
      (sparrow_count * x + swallow_count * y = 1)) →
    (4 * x + y = 5 * y + x ∧ 5 * x + 6 * y = 1) :=
by sorry

end sparrow_swallow_weight_system_l2104_210479


namespace bc_length_l2104_210481

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def is_right_triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define the lengths
def length (X Y : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem bc_length 
  (h1 : is_right_triangle A B C)
  (h2 : is_right_triangle A B D)
  (h3 : length A D = 50)
  (h4 : length C D = 25)
  (h5 : length A C = 20)
  (h6 : length A B = 15) :
  length B C = 25 :=
sorry

end bc_length_l2104_210481


namespace line_parallel_plane_theorem_l2104_210482

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the contained relation for lines and planes
variable (containedInPlane : Line → Plane → Prop)

-- Theorem statement
theorem line_parallel_plane_theorem 
  (a b : Line) (α : Plane) :
  parallelLine a b → parallelLinePlane a α →
  containedInPlane b α ∨ parallelLinePlane b α :=
sorry

end line_parallel_plane_theorem_l2104_210482


namespace carnation_percentage_l2104_210411

/-- Represents a flower bouquet with various types of flowers -/
structure Bouquet where
  total : ℝ
  pink_roses : ℝ
  red_roses : ℝ
  pink_carnations : ℝ
  red_carnations : ℝ
  yellow_tulips : ℝ

/-- Conditions for the flower bouquet problem -/
def bouquet_conditions (b : Bouquet) : Prop :=
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations + b.yellow_tulips = b.total ∧
  b.pink_roses + b.pink_carnations = 0.4 * b.total ∧
  b.red_roses + b.red_carnations = 0.4 * b.total ∧
  b.yellow_tulips = 0.2 * b.total ∧
  b.pink_roses = (2/5) * (b.pink_roses + b.pink_carnations) ∧
  b.red_carnations = (1/2) * (b.red_roses + b.red_carnations)

/-- Theorem stating that the percentage of carnations is 44% -/
theorem carnation_percentage (b : Bouquet) (h : bouquet_conditions b) : 
  (b.pink_carnations + b.red_carnations) / b.total = 0.44 := by
  sorry

end carnation_percentage_l2104_210411


namespace average_hamburgers_per_day_l2104_210416

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7

theorem average_hamburgers_per_day :
  (total_hamburgers : ℚ) / (days_in_week : ℚ) = 9 := by
  sorry

end average_hamburgers_per_day_l2104_210416


namespace square_less_than_four_times_l2104_210460

theorem square_less_than_four_times : ∀ n : ℤ, n^2 < 4*n ↔ n = 1 ∨ n = 2 ∨ n = 3 := by
  sorry

end square_less_than_four_times_l2104_210460


namespace solve_inequality_system_simplify_expression_l2104_210489

-- Part 1: System of inequalities
theorem solve_inequality_system (x : ℝ) :
  (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x ↔ 1 ≤ x ∧ x < 3 := by sorry

-- Part 2: Algebraic expression
theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (m - 1 / m) * (m^2 - m) / (m^2 - 2*m + 1) = m + 1 := by sorry

end solve_inequality_system_simplify_expression_l2104_210489


namespace total_spent_equals_sum_of_games_l2104_210425

/-- The total amount Tom spent on video games -/
def total_spent : ℝ := 35.52

/-- The cost of the football game -/
def football_cost : ℝ := 14.02

/-- The cost of the strategy game -/
def strategy_cost : ℝ := 9.46

/-- The cost of the Batman game -/
def batman_cost : ℝ := 12.04

/-- Theorem stating that the total amount spent is equal to the sum of individual game costs -/
theorem total_spent_equals_sum_of_games :
  total_spent = football_cost + strategy_cost + batman_cost := by
  sorry

end total_spent_equals_sum_of_games_l2104_210425


namespace sum_cos_fractions_24_pi_zero_l2104_210486

def simplest_proper_fractions_24 : List ℚ := [
  1/24, 5/24, 7/24, 11/24, 13/24, 17/24, 19/24, 23/24
]

theorem sum_cos_fractions_24_pi_zero : 
  (simplest_proper_fractions_24.map (fun x => Real.cos (x * Real.pi))).sum = 0 := by
  sorry

end sum_cos_fractions_24_pi_zero_l2104_210486


namespace not_in_S_iff_one_or_multiple_of_five_l2104_210429

def S : Set Nat := sorry

axiom two_in_S : 2 ∈ S

axiom square_in_S_implies_n_in_S : ∀ n : Nat, n^2 ∈ S → n ∈ S

axiom n_in_S_implies_n_plus_5_squared_in_S : ∀ n : Nat, n ∈ S → (n + 5)^2 ∈ S

axiom S_is_smallest : ∀ T : Set Nat, 
  (2 ∈ T ∧ 
   (∀ n : Nat, n^2 ∈ T → n ∈ T) ∧ 
   (∀ n : Nat, n ∈ T → (n + 5)^2 ∈ T)) → 
  S ⊆ T

theorem not_in_S_iff_one_or_multiple_of_five (n : Nat) :
  n ∉ S ↔ n = 1 ∨ ∃ k : Nat, n = 5 * k :=
sorry

end not_in_S_iff_one_or_multiple_of_five_l2104_210429


namespace min_value_expression_l2104_210401

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) :
  (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b ≥ Real.sqrt (8/3) := by
  sorry

end min_value_expression_l2104_210401


namespace min_seats_for_adjacency_l2104_210453

/-- Represents a row of seats -/
structure SeatRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if the next person must sit next to someone -/
def must_sit_next (row : SeatRow) : Prop :=
  ∀ i : ℕ, i < row.total_seats - 1 → (i % 4 = 0 → i < row.occupied_seats * 4)

/-- The main theorem to be proved -/
theorem min_seats_for_adjacency (row : SeatRow) :
  row.total_seats = 150 →
  (∀ r : SeatRow, r.total_seats = 150 → r.occupied_seats < 37 → ¬ must_sit_next r) →
  must_sit_next row →
  row.occupied_seats ≥ 37 :=
sorry

end min_seats_for_adjacency_l2104_210453


namespace max_min_values_on_interval_l2104_210458

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def additive (f : ℝ → ℝ) : Prop := ∀ x y, f (x + y) = f x + f y

theorem max_min_values_on_interval
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_additive : additive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_f1 : f 1 = -2) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ 6) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ -6) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 6) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -6) :=
sorry

end max_min_values_on_interval_l2104_210458


namespace intersection_nonempty_implies_m_range_l2104_210422

-- Define the sets A and B
def A (m : ℝ) := {x : ℝ | x^2 - 4*m*x + 2*m + 6 = 0}
def B := {x : ℝ | x < 0}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by sorry

end intersection_nonempty_implies_m_range_l2104_210422


namespace ball_radius_from_hole_dimensions_l2104_210407

/-- The radius of a spherical ball that leaves a circular hole in a frozen lake surface. -/
def ball_radius (hole_width : ℝ) (hole_depth : ℝ) : ℝ :=
  hole_depth

/-- Theorem stating that if a spherical ball leaves a hole 32 cm wide and 16 cm deep
    in a frozen lake surface, its radius is 16 cm. -/
theorem ball_radius_from_hole_dimensions :
  ball_radius 32 16 = 16 := by
  sorry

end ball_radius_from_hole_dimensions_l2104_210407


namespace team_pizza_consumption_l2104_210499

theorem team_pizza_consumption (total_slices : ℕ) (slices_left : ℕ) : 
  total_slices = 32 → slices_left = 7 → total_slices - slices_left = 25 := by
  sorry

end team_pizza_consumption_l2104_210499


namespace cassette_tape_cost_cassette_tape_cost_proof_l2104_210468

theorem cassette_tape_cost 
  (initial_amount : ℝ) 
  (headphone_cost : ℝ) 
  (num_tapes : ℕ) 
  (remaining_amount : ℝ) : ℝ :=
  let total_tape_cost := initial_amount - headphone_cost - remaining_amount
  total_tape_cost / num_tapes

#check cassette_tape_cost 50 25 2 7 = 9

theorem cassette_tape_cost_proof 
  (initial_amount : ℝ)
  (headphone_cost : ℝ)
  (num_tapes : ℕ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 50)
  (h2 : headphone_cost = 25)
  (h3 : num_tapes = 2)
  (h4 : remaining_amount = 7) :
  cassette_tape_cost initial_amount headphone_cost num_tapes remaining_amount = 9 := by
  sorry

end cassette_tape_cost_cassette_tape_cost_proof_l2104_210468


namespace rose_count_l2104_210462

theorem rose_count (total : ℕ) : 
  (300 ≤ total ∧ total ≤ 400) →
  (∃ x : ℕ, total = 21 * x + 13) →
  (∃ y : ℕ, total = 15 * y - 8) →
  total = 307 := by
sorry

end rose_count_l2104_210462


namespace power_equality_l2104_210443

theorem power_equality (q : ℕ) : 64^4 = 8^q → q = 8 := by
  sorry

end power_equality_l2104_210443


namespace total_people_in_program_l2104_210471

theorem total_people_in_program : 
  let parents : ℕ := 105
  let pupils : ℕ := 698
  let staff : ℕ := 45
  let performers : ℕ := 32
  parents + pupils + staff + performers = 880 := by
  sorry

end total_people_in_program_l2104_210471


namespace faulty_odometer_conversion_l2104_210445

/-- Represents an odometer that skips certain digits -/
structure FaultyOdometer where
  reading : Nat
  skipped_digits : List Nat

/-- Converts a faulty odometer reading to actual miles traveled -/
def actual_miles (odo : FaultyOdometer) : Nat :=
  sorry

/-- The theorem stating that a faulty odometer reading of 003006 
    (skipping 3 and 4) represents 1030 actual miles -/
theorem faulty_odometer_conversion :
  let odo : FaultyOdometer := { reading := 3006, skipped_digits := [3, 4] }
  actual_miles odo = 1030 := by
  sorry

end faulty_odometer_conversion_l2104_210445


namespace range_of_r_l2104_210484

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- State the theorem
theorem range_of_r :
  Set.range (fun x : ℝ => r x) = Set.Ici 9 := by
  sorry

end range_of_r_l2104_210484


namespace point_B_coordinates_l2104_210469

-- Define the vector type
def Vec := ℝ × ℝ

-- Define point A
def A : Vec := (-1, -5)

-- Define vector a
def a : Vec := (2, 3)

-- Define vector AB in terms of a
def AB : Vec := (3 * a.1, 3 * a.2)

-- Define point B
def B : Vec := (A.1 + AB.1, A.2 + AB.2)

-- Theorem statement
theorem point_B_coordinates : B = (5, 4) := by
  sorry

end point_B_coordinates_l2104_210469


namespace nested_expression_value_l2104_210434

/-- The nested expression that needs to be evaluated -/
def nestedExpression : ℕ := 2*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4)))))))))

/-- Theorem stating that the nested expression equals 699050 -/
theorem nested_expression_value : nestedExpression = 699050 := by
  sorry

end nested_expression_value_l2104_210434


namespace count_four_digit_divisible_by_13_l2104_210455

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 693 := by sorry

end count_four_digit_divisible_by_13_l2104_210455


namespace solution_to_system_l2104_210478

theorem solution_to_system (x y z : ℝ) :
  3 * (x^2 + y^2 + z^2) = 1 →
  x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3 →
  ((x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) :=
by sorry

end solution_to_system_l2104_210478


namespace batsman_average_increase_l2104_210437

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : Nat) : Rat :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 19) 
  (h2 : newAverage b 85 = b.average + 4) 
  (h3 : b.average > 0) : 
  newAverage b 85 = 9 := by
  sorry

end batsman_average_increase_l2104_210437


namespace ellipse_properties_l2104_210447

noncomputable section

-- Define the ellipse parameters
def a : ℝ := 2
def b : ℝ := Real.sqrt 3
def c : ℝ := 1

-- Define the eccentricity
def e : ℝ := 1 / 2

-- Define the maximum area of triangle PAB
def max_area : ℝ := 2 * Real.sqrt 3

-- Define the coordinates of point D
def d_x : ℝ := -11 / 8
def d_y : ℝ := 0

-- Define the constant dot product value
def constant_dot_product : ℝ := -135 / 64

-- Theorem statement
theorem ellipse_properties :
  (a > b ∧ b > 0) ∧
  (e = c / a) ∧
  (max_area = a * b) ∧
  (a^2 = b^2 + c^2) →
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∀ t : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧
    (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧
    (x₁ = t * y₁ - 1) ∧
    (x₂ = t * y₂ - 1) ∧
    ((x₁ - d_x) * (x₂ - d_x) + y₁ * y₂ = constant_dot_product)) :=
by sorry

end

end ellipse_properties_l2104_210447


namespace min_dist_point_on_line_l2104_210452

/-- The point that minimizes the sum of distances to two given points on a line -/
def minDistPoint (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The line 3x - 4y + 4 = 0 -/
def line (p : ℝ × ℝ) : Prop :=
  3 * p.1 - 4 * p.2 + 4 = 0

theorem min_dist_point_on_line :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (2, 15)
  let P : ℝ × ℝ := (8/3, 3)
  line P ∧
  ∀ Q : ℝ × ℝ, line Q →
    distance P A + distance P B ≤ distance Q A + distance Q B :=
sorry

end min_dist_point_on_line_l2104_210452


namespace sphere_cylinder_equal_area_l2104_210427

theorem sphere_cylinder_equal_area (r : ℝ) : 
  (4 * Real.pi * r^2 = 2 * Real.pi * 6 * 12) → r = 6 := by
  sorry

end sphere_cylinder_equal_area_l2104_210427


namespace max_baggies_of_cookies_l2104_210454

def chocolateChipCookies : ℕ := 23
def oatmealCookies : ℕ := 25
def cookiesPerBaggie : ℕ := 6

theorem max_baggies_of_cookies : 
  (chocolateChipCookies + oatmealCookies) / cookiesPerBaggie = 8 := by
  sorry

end max_baggies_of_cookies_l2104_210454


namespace smallest_integer_with_remainders_l2104_210448

theorem smallest_integer_with_remainders : ∃ b : ℕ+, 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ c : ℕ+, (c : ℕ) % 4 = 3 → (c : ℕ) % 6 = 5 → b ≤ c :=
by
  -- The proof goes here
  sorry

end smallest_integer_with_remainders_l2104_210448


namespace sum_of_zeros_is_four_l2104_210436

/-- Original parabola function -/
def original_parabola (x : ℝ) : ℝ := (x + 3)^2 - 2

/-- Transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 2)^2 + 2

/-- The zeros of the transformed parabola -/
def zeros : Set ℝ := {x | transformed_parabola x = 0}

theorem sum_of_zeros_is_four :
  ∃ (a b : ℝ), a ∈ zeros ∧ b ∈ zeros ∧ a + b = 4 := by
  sorry

end sum_of_zeros_is_four_l2104_210436


namespace polynomial_minimum_value_l2104_210433

theorem polynomial_minimum_value : 
  (∀ a b : ℝ, a^2 + 2*b^2 + 2*a + 4*b + 2008 ≥ 2005) ∧ 
  (∃ a b : ℝ, a^2 + 2*b^2 + 2*a + 4*b + 2008 = 2005) := by
  sorry

end polynomial_minimum_value_l2104_210433


namespace nell_initial_cards_l2104_210465

theorem nell_initial_cards (cards_given_to_jeff cards_left : ℕ) 
  (h1 : cards_given_to_jeff = 301)
  (h2 : cards_left = 154) :
  cards_given_to_jeff + cards_left = 455 :=
by sorry

end nell_initial_cards_l2104_210465


namespace trevors_age_problem_l2104_210418

theorem trevors_age_problem (trevor_age_decade_ago : ℕ) (brother_current_age : ℕ) : 
  trevor_age_decade_ago = 16 →
  brother_current_age = 32 →
  ∃ x : ℕ, x = 20 ∧ 2 * (trevor_age_decade_ago + 10 - x) = brother_current_age - x :=
by sorry

end trevors_age_problem_l2104_210418


namespace cistern_length_l2104_210494

theorem cistern_length (width : ℝ) (depth : ℝ) (wet_area : ℝ) (length : ℝ) : 
  width = 4 →
  depth = 1.25 →
  wet_area = 49 →
  wet_area = (length * width) + (2 * length * depth) + (2 * width * depth) →
  length = 6 := by
sorry

end cistern_length_l2104_210494


namespace initial_balloon_count_balloon_package_problem_l2104_210451

theorem initial_balloon_count (num_friends : ℕ) (balloons_given_back : ℕ) (final_balloons_per_friend : ℕ) : ℕ :=
  let initial_balloons_per_friend := final_balloons_per_friend + balloons_given_back
  num_friends * initial_balloons_per_friend

theorem balloon_package_problem :
  initial_balloon_count 5 11 39 = 250 := by
  sorry

end initial_balloon_count_balloon_package_problem_l2104_210451


namespace tank_plastering_cost_l2104_210459

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width height rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem stating the cost of plastering the specific tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

end tank_plastering_cost_l2104_210459


namespace isosceles_triangle_perimeter_l2104_210438

/-- An isosceles triangle with side lengths 3 and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 → b = 6 → c = 3 →
  (a = b ∨ a = c ∨ b = c) →  -- Isosceles condition
  a + b + c = 15 := by
  sorry

end isosceles_triangle_perimeter_l2104_210438


namespace total_books_proof_l2104_210439

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 19

/-- The number of books already read -/
def books_read : ℕ := 4

/-- The number of books yet to be read -/
def books_to_read : ℕ := 15

/-- Theorem stating that the total number of books is the sum of books read and books to be read -/
theorem total_books_proof : total_books = books_read + books_to_read := by
  sorry

end total_books_proof_l2104_210439


namespace third_day_income_l2104_210485

def cab_driver_problem (day1 day2 day3 day4 day5 : ℝ) : Prop :=
  day1 = 300 ∧ 
  day2 = 150 ∧ 
  day4 = 200 ∧ 
  day5 = 600 ∧ 
  (day1 + day2 + day3 + day4 + day5) / 5 = 400

theorem third_day_income (day1 day2 day3 day4 day5 : ℝ) 
  (h : cab_driver_problem day1 day2 day3 day4 day5) : day3 = 750 := by
  sorry

end third_day_income_l2104_210485


namespace cats_sold_during_sale_l2104_210408

/-- The number of cats sold during a pet store sale -/
theorem cats_sold_during_sale 
  (siamese_initial : ℕ) 
  (house_initial : ℕ) 
  (cats_left : ℕ) 
  (h1 : siamese_initial = 13)
  (h2 : house_initial = 5)
  (h3 : cats_left = 8) :
  siamese_initial + house_initial - cats_left = 10 :=
by sorry

end cats_sold_during_sale_l2104_210408


namespace sum_divisors_bound_l2104_210498

/-- σ(n) is the sum of the divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- ω(n) is the number of distinct prime divisors of n -/
def omega (n : ℕ+) : ℕ := sorry

/-- The sum of divisors of n is less than n multiplied by one more than
    the number of its distinct prime divisors -/
theorem sum_divisors_bound (n : ℕ+) : sigma n < n * (omega n + 1) := by sorry

end sum_divisors_bound_l2104_210498


namespace smallest_sum_M_N_l2104_210474

/-- Alice's transformation function -/
def aliceTransform (x : ℕ) : ℕ := 3 * x + 2

/-- Bob's transformation function -/
def bobTransform (x : ℕ) : ℕ := 2 * x + 27

/-- Alice's board after 4 moves -/
def aliceFourMoves (M : ℕ) : ℕ := aliceTransform (aliceTransform (aliceTransform (aliceTransform M)))

/-- Bob's board after 4 moves -/
def bobFourMoves (N : ℕ) : ℕ := bobTransform (bobTransform (bobTransform (bobTransform N)))

/-- The theorem stating the smallest sum of M and N -/
theorem smallest_sum_M_N : 
  ∃ (M N : ℕ), 
    M > 0 ∧ N > 0 ∧
    aliceFourMoves M = bobFourMoves N ∧
    (∀ (M' N' : ℕ), M' > 0 → N' > 0 → aliceFourMoves M' = bobFourMoves N' → M + N ≤ M' + N') ∧
    M + N = 10 := by
  sorry

end smallest_sum_M_N_l2104_210474


namespace min_value_and_inequality_l2104_210404

theorem min_value_and_inequality (a b x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hab : a + b = 1) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → a'^2 + b'^2/4 ≥ 1/5) ∧
  (a^2 + b^2/4 = 1/5 → a = 1/5 ∧ b = 4/5) ∧
  (a*x₁ + b*x₂) * (b*x₁ + a*x₂) ≥ x₁*x₂ := by
sorry

end min_value_and_inequality_l2104_210404


namespace square_root_equality_l2104_210415

theorem square_root_equality (m : ℝ) (x : ℝ) (h1 : m > 0) 
  (h2 : Real.sqrt m = x + 1) (h3 : Real.sqrt m = 5 + 2*x) : m = 1 := by
  sorry

end square_root_equality_l2104_210415


namespace train_speed_proof_l2104_210410

/-- The speed of the train from city A -/
def speed_train_A : ℝ := 60

/-- The speed of the train from city B -/
def speed_train_B : ℝ := 75

/-- The total distance between cities A and B in km -/
def total_distance : ℝ := 465

/-- The time in hours that the train from A travels before meeting -/
def time_train_A : ℝ := 4

/-- The time in hours that the train from B travels before meeting -/
def time_train_B : ℝ := 3

theorem train_speed_proof : 
  speed_train_A * time_train_A + speed_train_B * time_train_B = total_distance :=
by sorry

end train_speed_proof_l2104_210410


namespace arcsin_sqrt2_over_2_l2104_210480

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end arcsin_sqrt2_over_2_l2104_210480


namespace min_value_ab_l2104_210477

theorem min_value_ab (a b : ℝ) (h : 0 < a ∧ 0 < b) (eq : 1/a + 4/b = Real.sqrt (a*b)) : 
  4 ≤ a * b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 := by
  sorry

end min_value_ab_l2104_210477


namespace initial_concentration_proof_l2104_210496

/-- Proves that the initial concentration of a hydrochloric acid solution is 20%
    given the conditions of the problem. -/
theorem initial_concentration_proof (
  initial_amount : ℝ)
  (drained_amount : ℝ)
  (added_concentration : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_amount = 300)
  (h2 : drained_amount = 25)
  (h3 : added_concentration = 80 / 100)
  (h4 : final_concentration = 25 / 100)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 20 / 100 ∧
    (initial_amount - drained_amount) * initial_concentration +
    drained_amount * added_concentration =
    initial_amount * final_concentration :=
by sorry

end initial_concentration_proof_l2104_210496


namespace tory_fundraising_problem_l2104_210449

/-- Represents the fundraising problem for Tory's cookie sale --/
theorem tory_fundraising_problem (goal : ℕ) (chocolate_price oatmeal_price sugar_price : ℕ)
  (chocolate_sold oatmeal_sold sugar_sold : ℕ) :
  goal = 250 →
  chocolate_price = 6 →
  oatmeal_price = 5 →
  sugar_price = 4 →
  chocolate_sold = 5 →
  oatmeal_sold = 10 →
  sugar_sold = 15 →
  goal - (chocolate_price * chocolate_sold + oatmeal_price * oatmeal_sold + sugar_price * sugar_sold) = 110 := by
  sorry

#check tory_fundraising_problem

end tory_fundraising_problem_l2104_210449


namespace no_roots_implies_not_integer_l2104_210424

theorem no_roots_implies_not_integer (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) :
  ¬ ∃ n : ℤ, 20*(b - a) = n := by
  sorry

end no_roots_implies_not_integer_l2104_210424


namespace middle_school_run_time_average_l2104_210442

/-- Represents the average number of minutes run per day by students in a specific grade -/
structure GradeRunTime where
  grade : Nat
  average_minutes : ℝ

/-- Represents the ratio of students between two grades -/
structure GradeRatio where
  higher_grade : Nat
  lower_grade : Nat
  ratio : Nat

/-- Calculates the average run time for all students given the run times for each grade and the ratios between grades -/
def calculate_average_run_time (run_times : List GradeRunTime) (ratios : List GradeRatio) : ℝ :=
  sorry

theorem middle_school_run_time_average :
  let sixth_grade := GradeRunTime.mk 6 20
  let seventh_grade := GradeRunTime.mk 7 18
  let eighth_grade := GradeRunTime.mk 8 16
  let ratio_sixth_seventh := GradeRatio.mk 6 7 3
  let ratio_seventh_eighth := GradeRatio.mk 7 8 3
  let run_times := [sixth_grade, seventh_grade, eighth_grade]
  let ratios := [ratio_sixth_seventh, ratio_seventh_eighth]
  calculate_average_run_time run_times ratios = 250 / 13 := by
    sorry

end middle_school_run_time_average_l2104_210442


namespace condition_for_inequality_l2104_210423

theorem condition_for_inequality (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end condition_for_inequality_l2104_210423


namespace positive_correlation_implies_positive_slope_l2104_210461

/-- Represents a simple linear regression model --/
structure LinearRegression where
  b : ℝ  -- slope
  a : ℝ  -- y-intercept
  r : ℝ  -- correlation coefficient

/-- Theorem stating that a positive correlation coefficient implies a positive slope --/
theorem positive_correlation_implies_positive_slope (model : LinearRegression) :
  model.r > 0 → model.b > 0 := by
  sorry


end positive_correlation_implies_positive_slope_l2104_210461


namespace mn_perpendicular_pq_l2104_210466

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b : Point)

-- Define the quadrilateral and its properties
structure Quadrilateral : Type :=
  (A B C D : Point)
  (convex : Bool)

-- Define the intersection point of diagonals
def intersectionPoint (q : Quadrilateral) : Point :=
  sorry

-- Define centroid of a triangle
def centroid (p1 p2 p3 : Point) : Point :=
  sorry

-- Define orthocenter of a triangle
def orthocenter (p1 p2 p3 : Point) : Point :=
  sorry

-- Define perpendicularity of lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem mn_perpendicular_pq (q : Quadrilateral) :
  let O := intersectionPoint q
  let M := centroid q.A O q.B
  let N := centroid q.C O q.D
  let P := orthocenter q.B O q.C
  let Q := orthocenter q.D O q.A
  perpendicular (Line.mk M N) (Line.mk P Q) :=
sorry

end mn_perpendicular_pq_l2104_210466


namespace smallest_value_complex_sum_l2104_210456

theorem smallest_value_complex_sum (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_power : ω^4 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ (p q r : ℤ), p ≠ q ∧ q ≠ r ∧ p ≠ r → 
      Complex.abs (x + y*ω + z*ω^3) ≤ Complex.abs (p + q*ω + r*ω^3)) ∧
    Complex.abs (x + y*ω + z*ω^3) = 1 :=
sorry

end smallest_value_complex_sum_l2104_210456


namespace cube_surface_area_l2104_210475

theorem cube_surface_area (d : ℝ) (h : d = 8 * Real.sqrt 3) :
  6 * (d / Real.sqrt 3)^2 = 384 := by
  sorry

end cube_surface_area_l2104_210475


namespace convex_polygon_equal_area_division_l2104_210497

-- Define a convex polygon
structure ConvexPolygon where
  -- Add necessary properties to define a convex polygon
  is_convex : Bool

-- Define a line in 2D space
structure Line where
  -- Add necessary properties to define a line
  slope : ℝ
  intercept : ℝ

-- Define the concept of perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  -- Add condition for perpendicularity
  sorry

-- Define the concept of a region in the polygon
structure Region where
  -- Add necessary properties to define a region
  area : ℝ

-- Define the division of a polygon by two lines
def divide_polygon (p : ConvexPolygon) (l1 l2 : Line) : List Region :=
  -- Function to divide the polygon into regions
  sorry

-- Theorem statement
theorem convex_polygon_equal_area_division (p : ConvexPolygon) :
  ∃ (l1 l2 : Line), 
    perpendicular l1 l2 ∧ 
    let regions := divide_polygon p l1 l2
    regions.length = 4 ∧ 
    ∀ (r1 r2 : Region), r1 ∈ regions → r2 ∈ regions → r1.area = r2.area :=
by
  sorry

end convex_polygon_equal_area_division_l2104_210497


namespace tan_value_on_sqrt_graph_l2104_210406

/-- If the point (4, a) is on the graph of y = x^(1/2), then tan(aπ/6) = √3 -/
theorem tan_value_on_sqrt_graph (a : ℝ) : 
  a = 4^(1/2) → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end tan_value_on_sqrt_graph_l2104_210406
