import Mathlib

namespace proposition_equivalences_l2339_233992

theorem proposition_equivalences (a b c : ℝ) :
  (((c < 0 ∧ a * c > b * c) → a < b) ∧
   ((c < 0 ∧ a < b) → a * c > b * c) ∧
   ((c < 0 ∧ a * c ≤ b * c) → a ≥ b) ∧
   ((c < 0 ∧ a ≥ b) → a * c ≤ b * c) ∧
   ((a * b = 0) → (a = 0 ∨ b = 0)) ∧
   ((a = 0 ∨ b = 0) → a * b = 0) ∧
   ((a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)) ∧
   ((a ≠ 0 ∧ b ≠ 0) → a * b ≠ 0)) :=
by sorry

end proposition_equivalences_l2339_233992


namespace total_cost_of_pen_and_pencil_l2339_233962

def pencil_cost : ℝ := 2
def pen_cost : ℝ := pencil_cost + 9

theorem total_cost_of_pen_and_pencil :
  pencil_cost + pen_cost = 13 := by
  sorry

end total_cost_of_pen_and_pencil_l2339_233962


namespace waiter_problem_l2339_233934

theorem waiter_problem (initial_customers : ℕ) (left_customers : ℕ) (num_tables : ℕ) 
  (h1 : initial_customers = 62)
  (h2 : left_customers = 17)
  (h3 : num_tables = 5) :
  (initial_customers - left_customers) / num_tables = 9 := by
  sorry

end waiter_problem_l2339_233934


namespace anna_win_probability_l2339_233988

-- Define the game state as the sum modulo 4
inductive GameState
| Zero
| One
| Two
| Three

-- Define the die roll
def DieRoll : Type := Fin 6

-- Define the probability of winning for each game state
def winProbability : GameState → ℚ
| GameState.Zero => 0
| GameState.One => 50/99
| GameState.Two => 60/99
| GameState.Three => 62/99

-- Define the transition probability function
def transitionProbability (s : GameState) (r : DieRoll) : GameState :=
  match s, r.val + 1 with
  | GameState.Zero, n => match n % 4 with
    | 0 => GameState.Zero
    | 1 => GameState.One
    | 2 => GameState.Two
    | 3 => GameState.Three
    | _ => GameState.Zero  -- This case should never occur
  | GameState.One, n => match n % 4 with
    | 0 => GameState.One
    | 1 => GameState.Two
    | 2 => GameState.Three
    | 3 => GameState.Zero
    | _ => GameState.One  -- This case should never occur
  | GameState.Two, n => match n % 4 with
    | 0 => GameState.Two
    | 1 => GameState.Three
    | 2 => GameState.Zero
    | 3 => GameState.One
    | _ => GameState.Two  -- This case should never occur
  | GameState.Three, n => match n % 4 with
    | 0 => GameState.Three
    | 1 => GameState.Zero
    | 2 => GameState.One
    | 3 => GameState.Two
    | _ => GameState.Three  -- This case should never occur

-- Theorem statement
theorem anna_win_probability :
  (1 : ℚ) / 6 * (1 - winProbability GameState.Zero) +
  1 / 3 * (1 - winProbability GameState.One) +
  1 / 3 * (1 - winProbability GameState.Two) +
  1 / 6 * (1 - winProbability GameState.Three) = 52 / 99 :=
by sorry


end anna_win_probability_l2339_233988


namespace toucan_count_l2339_233922

/-- The number of toucans on the first limb initially -/
def initial_first_limb : ℕ := 3

/-- The number of toucans on the second limb initially -/
def initial_second_limb : ℕ := 4

/-- The number of toucans that join the first group -/
def join_first_limb : ℕ := 2

/-- The number of toucans that join the second group -/
def join_second_limb : ℕ := 3

/-- The total number of toucans after all changes -/
def total_toucans : ℕ := initial_first_limb + initial_second_limb + join_first_limb + join_second_limb

theorem toucan_count : total_toucans = 12 := by
  sorry

end toucan_count_l2339_233922


namespace sum_fractions_l2339_233901

theorem sum_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 2.4 := by
  sorry

end sum_fractions_l2339_233901


namespace gwen_spent_eight_dollars_l2339_233918

/-- The amount of money Gwen received for her birthday. -/
def initial_amount : ℕ := 14

/-- The amount of money Gwen has left. -/
def remaining_amount : ℕ := 6

/-- The amount of money Gwen spent. -/
def spent_amount : ℕ := initial_amount - remaining_amount

theorem gwen_spent_eight_dollars : spent_amount = 8 := by
  sorry

end gwen_spent_eight_dollars_l2339_233918


namespace inverse_sum_reciprocal_l2339_233998

theorem inverse_sum_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) := by
  sorry

end inverse_sum_reciprocal_l2339_233998


namespace arithmetic_expression_equality_l2339_233906

theorem arithmetic_expression_equality : 6 + 18 / 3 - 4 * 2 = 4 := by
  sorry

end arithmetic_expression_equality_l2339_233906


namespace ninth_power_negative_fourth_l2339_233977

theorem ninth_power_negative_fourth : (1 / 9)^(-1/4 : ℝ) = Real.sqrt 3 := by
  sorry

end ninth_power_negative_fourth_l2339_233977


namespace markup_percentage_is_45_l2339_233975

/-- Given a cost price, discount, and profit percentage, calculate the markup percentage. -/
def calculate_markup_percentage (cost_price discount : ℚ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage)
  let marked_price := selling_price + discount
  let markup := marked_price - cost_price
  (markup / cost_price) * 100

/-- Theorem: Given the specific values in the problem, the markup percentage is 45%. -/
theorem markup_percentage_is_45 :
  let cost_price : ℚ := 180
  let discount : ℚ := 45
  let profit_percentage : ℚ := 0.20
  calculate_markup_percentage cost_price discount profit_percentage = 45 := by
  sorry

#eval calculate_markup_percentage 180 45 0.20

end markup_percentage_is_45_l2339_233975


namespace possible_values_of_a_l2339_233935

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
sorry

end possible_values_of_a_l2339_233935


namespace intersection_and_chord_properties_l2339_233983

/-- Given two points M and N in a 2D Cartesian coordinate system -/
def M : ℝ × ℝ := (1, -3)
def N : ℝ × ℝ := (5, 1)

/-- Point C satisfies the given condition -/
def C (t : ℝ) : ℝ × ℝ :=
  (t * M.1 + (1 - t) * N.1, t * M.2 + (1 - t) * N.2)

/-- The parabola y^2 = 4x -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Perpendicularity of two vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem intersection_and_chord_properties :
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, C t = A) ∧ 
    (∃ t : ℝ, C t = B) ∧ 
    on_parabola A ∧ 
    on_parabola B ∧ 
    perpendicular A B) ∧
  (∃ P : ℝ × ℝ, P.1 = 4 ∧ P.2 = 0 ∧
    ∀ Q R : ℝ × ℝ, 
      on_parabola Q ∧ 
      on_parabola R ∧ 
      (Q.2 - P.2) * (R.1 - P.1) = (Q.1 - P.1) * (R.2 - P.2) →
      (Q.1 * R.1 + Q.2 * R.2 = 0)) :=
sorry

end intersection_and_chord_properties_l2339_233983


namespace coefficient_x3y7_expansion_l2339_233917

theorem coefficient_x3y7_expansion :
  let n : ℕ := 10
  let k : ℕ := 3
  let coeff : ℚ := (n.choose k) * (2/3)^k * (-3/5)^(n-k)
  coeff = -256/257 := by sorry

end coefficient_x3y7_expansion_l2339_233917


namespace sams_first_month_earnings_l2339_233980

/-- Sam's hourly rate for Math tutoring -/
def hourly_rate : ℕ := 10

/-- The difference in earnings between the second and first month -/
def second_month_increase : ℕ := 150

/-- Total hours spent tutoring over two months -/
def total_hours : ℕ := 55

/-- Sam's earnings in the first month -/
def first_month_earnings : ℕ := 200

/-- Theorem stating that Sam's earnings in the first month were $200 -/
theorem sams_first_month_earnings :
  first_month_earnings = (hourly_rate * total_hours - second_month_increase) / 2 :=
by sorry

end sams_first_month_earnings_l2339_233980


namespace circle_passes_through_fixed_point_circle_tangent_conditions_l2339_233939

/-- The equation of the given circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*(a - 1) = 0

/-- The equation of the fixed circle -/
def fixed_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_passes_through_fixed_point :
  ∀ a : ℝ, circle_equation 4 (-2) a := by sorry

theorem circle_tangent_conditions :
  ∀ a : ℝ, (∃ x y : ℝ, circle_equation x y a ∧ fixed_circle_equation x y ∧
    (∀ x' y' : ℝ, circle_equation x' y' a ∧ fixed_circle_equation x' y' → (x', y') = (x, y))) ↔
  (a = 1 - Real.sqrt 5 ∨ a = 1 + Real.sqrt 5) := by sorry

end circle_passes_through_fixed_point_circle_tangent_conditions_l2339_233939


namespace inequality_proof_l2339_233978

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + d))) + (1 / (d * (1 + a))) ≥ 2 := by
  sorry

end inequality_proof_l2339_233978


namespace exponent_and_polynomial_identities_l2339_233955

variable (a b : ℝ)

theorem exponent_and_polynomial_identities : 
  ((a^2)^3 / (-a)^2 = a^4) ∧ 
  ((a+2*b)*(a+b)-3*a*(a+b) = -2*a^2 + 2*b^2) := by sorry

end exponent_and_polynomial_identities_l2339_233955


namespace total_tickets_sold_l2339_233982

/-- Given the number of tickets sold in section A and section B, prove that the total number of tickets sold is their sum. -/
theorem total_tickets_sold (section_a_tickets : ℕ) (section_b_tickets : ℕ) :
  section_a_tickets = 2900 →
  section_b_tickets = 1600 →
  section_a_tickets + section_b_tickets = 4500 := by
  sorry

end total_tickets_sold_l2339_233982


namespace orange_juice_percentage_l2339_233948

theorem orange_juice_percentage (total : ℝ) (watermelon_percent : ℝ) (grape_ounces : ℝ) :
  total = 140 →
  watermelon_percent = 60 →
  grape_ounces = 35 →
  (15 : ℝ) / 100 * total = total - watermelon_percent / 100 * total - grape_ounces :=
by sorry

end orange_juice_percentage_l2339_233948


namespace min_ping_pong_balls_l2339_233915

def is_valid_box_count (n : ℕ) : Prop :=
  n ≥ 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_counts (counts : List ℕ) : Prop :=
  counts.Nodup

theorem min_ping_pong_balls :
  ∃ (counts : List ℕ),
    counts.length = 10 ∧
    (∀ n ∈ counts, is_valid_box_count n) ∧
    distinct_counts counts ∧
    counts.sum = 174 ∧
    (∀ (other_counts : List ℕ),
      other_counts.length = 10 →
      (∀ n ∈ other_counts, is_valid_box_count n) →
      distinct_counts other_counts →
      other_counts.sum ≥ 174) :=
by sorry

end min_ping_pong_balls_l2339_233915


namespace pascal_sum_29_l2339_233940

/-- Number of elements in a row of Pascal's Triangle -/
def pascal_row_count (n : ℕ) : ℕ := n + 1

/-- Sum of elements in Pascal's Triangle from row 0 to row n -/
def pascal_sum (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2

theorem pascal_sum_29 : pascal_sum 29 = 465 := by
  sorry

end pascal_sum_29_l2339_233940


namespace inscribed_rectangle_epsilon_l2339_233921

-- Define the triangle
structure Triangle :=
  (MN NP PM : ℝ)

-- Define the rectangle
structure Rectangle :=
  (W X Y Z : ℝ × ℝ)

-- Define the area function
def rectangleArea (γ ε δ : ℝ) : ℝ := γ * δ - δ * ε^2

theorem inscribed_rectangle_epsilon (t : Triangle) (r : Rectangle) (γ ε : ℝ) :
  t.MN = 10 ∧ t.NP = 24 ∧ t.PM = 26 →
  (∃ δ, rectangleArea γ ε δ = 0) →
  (∃ δ, rectangleArea γ ε δ = 60) →
  ε = 5/12 := by
  sorry

#check inscribed_rectangle_epsilon

end inscribed_rectangle_epsilon_l2339_233921


namespace all_permissible_triangles_in_final_set_l2339_233932

/-- A permissible triangle for a prime p is represented by its angles as multiples of (180/p) degrees -/
structure PermissibleTriangle (p : ℕ) :=
  (a b c : ℕ)
  (sum_eq_p : a + b + c = p)
  (p_prime : Nat.Prime p)

/-- The set of all permissible triangles for a given prime p -/
def allPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | true}

/-- A function that represents cutting a triangle into two different permissible triangles -/
def cutTriangle (p : ℕ) (t : PermissibleTriangle p) : Option (PermissibleTriangle p × PermissibleTriangle p) :=
  sorry

/-- The set of triangles resulting from repeated cutting until no more cuts are possible -/
def finalTriangleSet (p : ℕ) (initial : PermissibleTriangle p) : Set (PermissibleTriangle p) :=
  sorry

/-- The main theorem: the final set of triangles includes all possible permissible triangles -/
theorem all_permissible_triangles_in_final_set (p : ℕ) (hp : Nat.Prime p) (initial : PermissibleTriangle p) :
  finalTriangleSet p initial = allPermissibleTriangles p :=
sorry

end all_permissible_triangles_in_final_set_l2339_233932


namespace exponent_equality_l2339_233914

theorem exponent_equality (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponent_equality_l2339_233914


namespace star_two_three_l2339_233913

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 * b^2 - a + 1

-- Theorem statement
theorem star_two_three : star 2 3 = 35 := by
  sorry

end star_two_three_l2339_233913


namespace cosine_double_angle_equation_cosine_double_angle_special_case_l2339_233970

theorem cosine_double_angle_equation (a b c : ℝ) (x : ℝ) 
  (h : a * (Real.cos x)^2 + b * Real.cos x + c = 0) :
  (1/4) * a^2 * (Real.cos (2*x))^2 + 
  (1/2) * (a^2 - b^2 + 2*a*c) * Real.cos (2*x) + 
  (1/4) * (a^2 + 4*a*c + 4*c^2 - 2*b^2) = 0 := by
  sorry

-- Special case
theorem cosine_double_angle_special_case (x : ℝ) 
  (h : 4 * (Real.cos x)^2 + 2 * Real.cos x - 1 = 0) :
  4 * (Real.cos (2*x))^2 + 2 * Real.cos (2*x) - 1 = 0 := by
  sorry

end cosine_double_angle_equation_cosine_double_angle_special_case_l2339_233970


namespace infinite_non_sum_of_three_cubes_l2339_233905

theorem infinite_non_sum_of_three_cubes :
  ∀ k : ℤ, ¬∃ a b c : ℤ, (9*k + 4 = a^3 + b^3 + c^3) ∧ ¬∃ a b c : ℤ, (9*k - 4 = a^3 + b^3 + c^3) :=
by
  sorry

end infinite_non_sum_of_three_cubes_l2339_233905


namespace actual_miles_traveled_l2339_233941

/-- A function that counts the number of integers from 0 to n (inclusive) that contain the digit 3 --/
def countWithThree (n : ℕ) : ℕ := sorry

/-- The odometer reading --/
def odometerReading : ℕ := 3008

/-- Theorem stating that the actual miles traveled is 2465 when the odometer reads 3008 --/
theorem actual_miles_traveled :
  odometerReading - countWithThree odometerReading = 2465 := by sorry

end actual_miles_traveled_l2339_233941


namespace dad_steps_l2339_233930

/-- Represents the number of steps taken by each person --/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad, Masha, and Yasha --/
def step_relation (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha ∧ 5 * s.masha = 3 * s.yasha

/-- The total number of steps taken by Masha and Yasha --/
def total_masha_yasha (s : Steps) : ℕ := s.masha + s.yasha

/-- Theorem stating that given the conditions, Dad took 90 steps --/
theorem dad_steps :
  ∀ s : Steps,
  step_relation s →
  total_masha_yasha s = 400 →
  s.dad = 90 :=
by
  sorry


end dad_steps_l2339_233930


namespace weight_of_replaced_person_l2339_233950

theorem weight_of_replaced_person (initial_count : ℕ) (avg_increase : ℚ) (new_weight : ℚ) :
  initial_count = 6 →
  avg_increase = 4.5 →
  new_weight = 102 →
  ∃ (old_weight : ℚ), old_weight = 75 ∧ new_weight = old_weight + initial_count * avg_increase :=
by sorry

end weight_of_replaced_person_l2339_233950


namespace number_of_cat_only_owners_cat_only_owners_count_l2339_233953

theorem number_of_cat_only_owners (total_pet_owners : ℕ) (only_dog_owners : ℕ) 
  (cat_and_dog_owners : ℕ) (cat_dog_snake_owners : ℕ) (total_snakes : ℕ) : ℕ :=
  let snake_only_owners := total_snakes - cat_dog_snake_owners
  let cat_only_owners := total_pet_owners - only_dog_owners - cat_and_dog_owners - 
                         cat_dog_snake_owners - snake_only_owners
  cat_only_owners

theorem cat_only_owners_count : 
  number_of_cat_only_owners 69 15 5 3 39 = 10 := by
  sorry

end number_of_cat_only_owners_cat_only_owners_count_l2339_233953


namespace y_intercept_of_line_l2339_233966

/-- The y-intercept of a line with slope 1 passing through the midpoint of a line segment --/
theorem y_intercept_of_line (x₁ y₁ x₂ y₂ : ℝ) :
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  let slope := 1
  let y_intercept := midpoint_y - slope * midpoint_x
  x₁ = 2 ∧ y₁ = 8 ∧ x₂ = 14 ∧ y₂ = 4 →
  y_intercept = -2 := by
sorry

end y_intercept_of_line_l2339_233966


namespace sand_bag_cost_l2339_233987

/-- The cost of a bag of sand given the dimensions of a square sandbox,
    the area covered by one bag, and the total cost to fill the sandbox. -/
theorem sand_bag_cost
  (sandbox_side : ℝ)
  (bag_area : ℝ)
  (total_cost : ℝ)
  (h_square : sandbox_side = 3)
  (h_bag : bag_area = 3)
  (h_cost : total_cost = 12) :
  total_cost / (sandbox_side ^ 2 / bag_area) = 4 := by
sorry

end sand_bag_cost_l2339_233987


namespace abs_diff_bound_l2339_233961

theorem abs_diff_bound (a b c h : ℝ) (ha : |a - c| < h) (hb : |b - c| < h) : |a - b| < 2 * h := by
  sorry

end abs_diff_bound_l2339_233961


namespace line_equation_l2339_233924

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- Point M -/
def M : ℝ × ℝ := (4, 1)

/-- Line passing through two points -/
def line_through (p₁ p₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p₁.2) * (p₂.1 - p₁.1) = (x - p₁.1) * (p₂.2 - p₁.2)

/-- Midpoint of two points -/
def is_midpoint (m p₁ p₂ : ℝ × ℝ) : Prop :=
  m.1 = (p₁.1 + p₂.1) / 2 ∧ m.2 = (p₁.2 + p₂.2) / 2

theorem line_equation :
  ∃ (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    is_midpoint M A B ∧
    (∀ x y, line_through M (x, y) x y ↔ y = 8*x - 31) :=
by sorry

end line_equation_l2339_233924


namespace pie_chart_statement_is_false_l2339_233904

-- Define the characteristics of different chart types
def BarChart : Type := Unit
def LineChart : Type := Unit
def PieChart : Type := Unit

-- Define what each chart type can represent
def represents_amount (chart : Type) : Prop := sorry
def represents_changes (chart : Type) : Prop := sorry
def represents_part_whole (chart : Type) : Prop := sorry

-- State the known characteristics of each chart type
axiom bar_chart_amount : represents_amount BarChart
axiom line_chart_amount_and_changes : represents_amount LineChart ∧ represents_changes LineChart
axiom pie_chart_part_whole : represents_part_whole PieChart

-- The statement we want to prove false
def pie_chart_statement : Prop :=
  represents_amount PieChart ∧ represents_changes PieChart

-- The theorem to prove
theorem pie_chart_statement_is_false : ¬pie_chart_statement := by
  sorry

end pie_chart_statement_is_false_l2339_233904


namespace bullying_instances_l2339_233984

def days_per_bullying : ℕ := 3
def typical_fingers_and_toes : ℕ := 20
def additional_suspension_days : ℕ := 14

def total_suspension_days : ℕ := 3 * typical_fingers_and_toes + additional_suspension_days

theorem bullying_instances : 
  (total_suspension_days / days_per_bullying : ℕ) = 24 := by
  sorry

end bullying_instances_l2339_233984


namespace equal_angles_with_perpendicular_circle_l2339_233991

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (passes_through : Circle → Point → Prop)
variable (tangent_to : Circle → Circle → Prop)
variable (perpendicular_to : Circle → Circle → Prop)
variable (angle_between : Circle → Circle → ℝ)

-- State the theorem
theorem equal_angles_with_perpendicular_circle
  (A B : Point) (S S₁ S₂ S₃ : Circle)
  (h1 : passes_through S₁ A ∧ passes_through S₁ B)
  (h2 : passes_through S₂ A ∧ passes_through S₂ B)
  (h3 : tangent_to S₁ S)
  (h4 : tangent_to S₂ S)
  (h5 : perpendicular_to S₃ S) :
  angle_between S₃ S₁ = angle_between S₃ S₂ :=
by sorry

end equal_angles_with_perpendicular_circle_l2339_233991


namespace car_speed_problem_l2339_233933

/-- Given two cars traveling on a 500-mile highway from opposite ends, 
    one at speed v and the other at 60 mph, meeting after 5 hours, 
    prove that the speed v of the first car is 40 mph. -/
theorem car_speed_problem (v : ℝ) 
  (h1 : v > 0) -- Assuming speed is positive
  (h2 : 5 * v + 5 * 60 = 500) : v = 40 := by
  sorry

end car_speed_problem_l2339_233933


namespace reciprocal_of_negative_three_l2339_233947

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end reciprocal_of_negative_three_l2339_233947


namespace factory_output_equation_l2339_233937

/-- Represents the factory's output model -/
def factory_output (initial_output : ℝ) (growth_rate : ℝ) (months : ℕ) : ℝ :=
  initial_output * (1 + growth_rate) ^ months

/-- Theorem stating that the equation 500(1+x)^2 = 720 correctly represents the factory's output in March -/
theorem factory_output_equation (x : ℝ) : 
  factory_output 500 x 2 = 720 ↔ 500 * (1 + x)^2 = 720 := by
  sorry

end factory_output_equation_l2339_233937


namespace simplify_expression_l2339_233903

theorem simplify_expression (m : ℝ) : (3*m + 2) - 3*(m^2 - m + 1) + (3 - 6*m) = -3*m^2 + 2 := by
  sorry

end simplify_expression_l2339_233903


namespace limit_evaluation_l2339_233943

theorem limit_evaluation : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((1 + 2*n : ℝ)^3 - 8*n^5) / ((1 + 2*n)^2 + 4*n^2) + 1| < ε :=
by sorry

end limit_evaluation_l2339_233943


namespace emily_commute_time_l2339_233910

/-- Calculates the total commute time for Emily given her travel distances and local road time --/
theorem emily_commute_time 
  (freeway_distance : ℝ) 
  (local_distance : ℝ) 
  (local_time : ℝ) 
  (h1 : freeway_distance = 100) 
  (h2 : local_distance = 25) 
  (h3 : local_time = 50) 
  (h4 : freeway_distance / local_distance = 4) : 
  local_time + freeway_distance / (2 * local_distance / local_time) = 150 := by
  sorry

#check emily_commute_time

end emily_commute_time_l2339_233910


namespace sphere_surface_area_l2339_233907

theorem sphere_surface_area (diameter : ℝ) (h : diameter = 10) :
  4 * Real.pi * (diameter / 2)^2 = 100 * Real.pi := by
  sorry

end sphere_surface_area_l2339_233907


namespace triangle_angle_A_l2339_233926

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = 4 →
  Real.sin B = 2/3 →
  a < b →
  (Real.sin A) * b = a * (Real.sin B) →
  A = π/6 := by
  sorry

end triangle_angle_A_l2339_233926


namespace parallelogram_base_l2339_233938

theorem parallelogram_base (height area : ℝ) (h1 : height = 32) (h2 : area = 896) : 
  area / height = 28 := by
  sorry

end parallelogram_base_l2339_233938


namespace shipping_cost_correct_l2339_233927

/-- The cost function for shipping packages -/
def shipping_cost (W : ℕ) : ℝ :=
  5 + 4 * (W - 1)

/-- Theorem stating the correctness of the shipping cost formula -/
theorem shipping_cost_correct (W : ℕ) (h : W ≥ 2) :
  shipping_cost W = 5 + 4 * (W - 1) :=
by sorry

end shipping_cost_correct_l2339_233927


namespace min_trees_for_three_types_l2339_233995

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset ℕ)
  (type : ℕ → TreeType)
  (total_trees : trees.card = 100)
  (four_types_in_85 : ∀ s : Finset ℕ, s ⊆ trees → s.card = 85 → 
    (∃ i ∈ s, type i = TreeType.Birch) ∧
    (∃ i ∈ s, type i = TreeType.Spruce) ∧
    (∃ i ∈ s, type i = TreeType.Pine) ∧
    (∃ i ∈ s, type i = TreeType.Aspen))

/-- The main theorem stating the minimum number of trees to guarantee at least three types -/
theorem min_trees_for_three_types (g : Grove) :
  ∀ s : Finset ℕ, s ⊆ g.trees → s.card ≥ 69 →
    (∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
      (∃ i ∈ s, g.type i = t1) ∧
      (∃ i ∈ s, g.type i = t2) ∧
      (∃ i ∈ s, g.type i = t3)) :=
by sorry

end min_trees_for_three_types_l2339_233995


namespace sin_600_degrees_l2339_233929

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_degrees_l2339_233929


namespace square_side_length_l2339_233979

theorem square_side_length (r : ℝ) (h : r = 3) : 
  ∃ x : ℝ, 4 * x = 2 * π * r ∧ x = 3 * π / 2 := by
  sorry

end square_side_length_l2339_233979


namespace exist_six_consecutive_naturals_lcm_property_l2339_233942

theorem exist_six_consecutive_naturals_lcm_property :
  ∃ n : ℕ, lcm (lcm n (n + 1)) (n + 2) > lcm (lcm (n + 3) (n + 4)) (n + 5) := by
  sorry

end exist_six_consecutive_naturals_lcm_property_l2339_233942


namespace contrapositive_theorem_l2339_233993

theorem contrapositive_theorem (x : ℝ) :
  (x = 1 ∨ x = 2 → x^2 - 3*x + 2 ≤ 0) ↔ (x^2 - 3*x + 2 > 0 → x ≠ 1 ∧ x ≠ 2) :=
by sorry

end contrapositive_theorem_l2339_233993


namespace no_rain_probability_l2339_233959

theorem no_rain_probability (p : ℝ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end no_rain_probability_l2339_233959


namespace nested_square_root_equality_l2339_233964

theorem nested_square_root_equality : 
  Real.sqrt (1 + 2014 * Real.sqrt (1 + 2015 * Real.sqrt (1 + 2016 * 2018))) = 2015 := by
  sorry

end nested_square_root_equality_l2339_233964


namespace union_of_M_and_N_l2339_233957

def M : Set ℕ := {1, 2}

def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end union_of_M_and_N_l2339_233957


namespace consecutive_integers_fourth_power_sum_l2339_233954

theorem consecutive_integers_fourth_power_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) = 12 * (3 * n + 3) → 
  n^4 + (n + 1)^4 + (n + 2)^4 = 7793 := by
sorry

end consecutive_integers_fourth_power_sum_l2339_233954


namespace pentagon_angle_measure_l2339_233920

-- Define a pentagon
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

-- Define the theorem
theorem pentagon_angle_measure (PQRST : Pentagon) 
  (h1 : PQRST.P = PQRST.R ∧ PQRST.R = PQRST.T)  -- ∠P ≅ ∠R ≅ ∠T
  (h2 : PQRST.Q + PQRST.S = 180)  -- ∠Q is supplementary to ∠S
  (h3 : PQRST.P + PQRST.Q + PQRST.R + PQRST.S + PQRST.T = 540)  -- Sum of angles in a pentagon
  : PQRST.T = 120 := by
  sorry

end pentagon_angle_measure_l2339_233920


namespace missing_digit_divisible_by_three_l2339_233997

theorem missing_digit_divisible_by_three (x : Nat) :
  (x < 10) →
  (246 * 100 + x * 10 + 9) % 3 = 0 →
  x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
sorry

end missing_digit_divisible_by_three_l2339_233997


namespace max_candy_pieces_l2339_233936

theorem max_candy_pieces (n : ℕ) (μ : ℚ) (min_pieces : ℕ) : 
  n = 35 → 
  μ = 6 → 
  min_pieces = 2 →
  ∃ (max_pieces : ℕ), 
    max_pieces = 142 ∧ 
    (∀ (student_pieces : List ℕ), 
      student_pieces.length = n ∧ 
      (∀ x ∈ student_pieces, x ≥ min_pieces) ∧ 
      (student_pieces.sum : ℚ) / n = μ →
      ∀ x ∈ student_pieces, x ≤ max_pieces) :=
by sorry

end max_candy_pieces_l2339_233936


namespace garage_motorcycles_l2339_233952

theorem garage_motorcycles (total_wheels : ℕ) (bicycles : ℕ) (cars : ℕ) 
  (bicycle_wheels : ℕ) (car_wheels : ℕ) (motorcycle_wheels : ℕ) :
  total_wheels = 90 ∧ 
  bicycles = 20 ∧ 
  cars = 10 ∧ 
  bicycle_wheels = 2 ∧ 
  car_wheels = 4 ∧ 
  motorcycle_wheels = 2 → 
  (total_wheels - (bicycles * bicycle_wheels + cars * car_wheels)) / motorcycle_wheels = 5 :=
by sorry

end garage_motorcycles_l2339_233952


namespace married_men_count_l2339_233946

theorem married_men_count (total : ℕ) (tv : ℕ) (radio : ℕ) (ac : ℕ) (all_and_married : ℕ) 
  (h_total : total = 100)
  (h_tv : tv = 75)
  (h_radio : radio = 85)
  (h_ac : ac = 70)
  (h_all_and_married : all_and_married = 12)
  (h_all_and_married_le_total : all_and_married ≤ total) :
  ∃ (married : ℕ), married ≥ all_and_married ∧ married ≤ total :=
by
  sorry

end married_men_count_l2339_233946


namespace sequence_is_arithmetic_l2339_233976

/-- Given a sequence {aₙ} satisfying 4aₙ₊₁ - 4aₙ - 9 = 0 for all n,
    prove that {aₙ} is an arithmetic sequence with a common difference of 9/4. -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) 
    (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
    ∃ d, d = 9/4 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end sequence_is_arithmetic_l2339_233976


namespace defective_units_percentage_l2339_233949

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0036) :
  ∃ (defective_ratio : Real),
    defective_ratio * shipped_defective_ratio = total_shipped_defective_ratio ∧
    defective_ratio = 0.09 := by
  sorry

end defective_units_percentage_l2339_233949


namespace regular_hexagon_angles_l2339_233999

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 angles of equal measure. -/
structure RegularHexagon where
  -- We don't need to define any specific fields for this problem

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_measure (h : RegularHexagon) : ℝ := 120

/-- The sum of all exterior angles of a regular hexagon -/
def sum_exterior_angles (h : RegularHexagon) : ℝ := 360

theorem regular_hexagon_angles (h : RegularHexagon) : 
  (interior_angle_measure h = 120) ∧ (sum_exterior_angles h = 360) := by
  sorry

#check regular_hexagon_angles

end regular_hexagon_angles_l2339_233999


namespace sum_and_ratio_to_difference_l2339_233911

theorem sum_and_ratio_to_difference (x y : ℚ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 4/5) :
  y - x = 500/9 := by
sorry

end sum_and_ratio_to_difference_l2339_233911


namespace a_zero_sufficient_not_necessary_l2339_233909

def M (a : ℝ) : Set ℝ := {1, a}
def N : Set ℝ := {-1, 0, 1}

theorem a_zero_sufficient_not_necessary (a : ℝ) :
  (a = 0 → M a ⊆ N) ∧ ¬(M a ⊆ N → a = 0) :=
sorry

end a_zero_sufficient_not_necessary_l2339_233909


namespace cubic_inequality_l2339_233960

theorem cubic_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end cubic_inequality_l2339_233960


namespace trig_identity_l2339_233994

/-- Prove that (cos 70° * cos 20°) / (1 - 2 * sin² 25°) = 1/2 -/
theorem trig_identity : 
  (Real.cos (70 * π / 180) * Real.cos (20 * π / 180)) / 
  (1 - 2 * Real.sin (25 * π / 180) ^ 2) = 1/2 := by
  sorry

end trig_identity_l2339_233994


namespace fathers_age_fathers_current_age_l2339_233958

theorem fathers_age (sons_age_next_year : ℕ) (father_age_ratio : ℕ) : ℕ :=
  let sons_current_age := sons_age_next_year - 1
  father_age_ratio * sons_current_age

theorem fathers_current_age :
  fathers_age 8 5 = 35 := by
  sorry

end fathers_age_fathers_current_age_l2339_233958


namespace paper_folding_holes_l2339_233974

/-- The number of small squares along each side after folding a square paper n times -/
def squares_per_side (n : ℕ) : ℕ := 2^n

/-- The number of internal edges along each side after folding -/
def internal_edges (n : ℕ) : ℕ := squares_per_side n - 1

/-- The total number of holes in the middle of the paper after folding n times -/
def total_holes (n : ℕ) : ℕ := internal_edges n * squares_per_side n

/-- Theorem: When a square piece of paper is folded in half 6 times and a notch is cut along
    each edge of the resulting small square, the number of small holes in the middle
    when unfolded is 4032. -/
theorem paper_folding_holes :
  total_holes 6 = 4032 := by sorry

end paper_folding_holes_l2339_233974


namespace average_of_remaining_numbers_l2339_233900

theorem average_of_remaining_numbers
  (total : ℝ) (group1 : ℝ) (group2 : ℝ) (group3 : ℝ)
  (h1 : total = 6 * 3.95)
  (h2 : group1 = 2 * 4.4)
  (h3 : group2 = 2 * 3.85)
  (h4 : group3 = total - (group1 + group2)) :
  group3 / 2 = 3.6 := by
sorry

end average_of_remaining_numbers_l2339_233900


namespace scale_division_l2339_233919

theorem scale_division (total_length : ℝ) (num_parts : ℕ) (part_length : ℝ) : 
  total_length = 90 → num_parts = 5 → part_length * num_parts = total_length → part_length = 18 := by
  sorry

end scale_division_l2339_233919


namespace work_completion_time_l2339_233931

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a = 4 → b = 12 → 1 / (1 / a + 1 / b) = 3 := by
  sorry

end work_completion_time_l2339_233931


namespace opposite_sides_parameter_set_is_correct_l2339_233972

/-- The set of parameter values for which points A and B lie on opposite sides of a line -/
def opposite_sides_parameter_set : Set ℝ :=
  {a | a < -2 ∨ (0 < a ∧ a < 2/3) ∨ a > 8/7}

/-- Equation of point A -/
def point_A_eq (a x y : ℝ) : Prop :=
  5 * a^2 + 12 * a * x + 4 * a * y + 8 * x^2 + 8 * x * y + 4 * y^2 = 0

/-- Equation of parabola with vertex at point B -/
def parabola_B_eq (a x y : ℝ) : Prop :=
  a * x^2 - 2 * a^2 * x - a * y + a^3 + 4 = 0

/-- Equation of the line -/
def line_eq (x y : ℝ) : Prop :=
  y - 3 * x = 4

/-- Theorem stating that the set of parameter values is correct -/
theorem opposite_sides_parameter_set_is_correct :
  ∀ a : ℝ, a ∈ opposite_sides_parameter_set ↔
    ∃ (x_A y_A x_B y_B : ℝ),
      point_A_eq a x_A y_A ∧
      parabola_B_eq a x_B y_B ∧
      ¬line_eq x_A y_A ∧
      ¬line_eq x_B y_B ∧
      (line_eq x_A y_A ↔ ¬line_eq x_B y_B) :=
by sorry

end opposite_sides_parameter_set_is_correct_l2339_233972


namespace uncle_zhang_age_uncle_zhang_age_proof_l2339_233908

theorem uncle_zhang_age : Nat → Nat → Prop :=
  fun zhang_age li_age =>
    zhang_age + li_age = 56 ∧
    2 * (li_age - (li_age - zhang_age)) = li_age ∧
    zhang_age = 24

-- The proof is omitted
theorem uncle_zhang_age_proof : ∃ (zhang_age li_age : Nat), uncle_zhang_age zhang_age li_age :=
  sorry

end uncle_zhang_age_uncle_zhang_age_proof_l2339_233908


namespace elf_goblin_theorem_l2339_233956

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid arrangements of elves and goblins -/
def elf_goblin_arrangements (n : ℕ) : ℕ := fib (n + 1)

/-- Theorem: The number of valid arrangements of n elves and n goblins,
    where no two goblins can be adjacent, is equal to the (n+2)th Fibonacci number -/
theorem elf_goblin_theorem (n : ℕ) :
  elf_goblin_arrangements n = fib (n + 1) :=
by sorry

end elf_goblin_theorem_l2339_233956


namespace ln_inequality_l2339_233928

theorem ln_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < Real.exp 1) :
  a * Real.log b > b * Real.log a := by
  sorry

end ln_inequality_l2339_233928


namespace duck_count_l2339_233990

theorem duck_count (total_animals : ℕ) (total_legs : ℕ) (duck_legs : ℕ) (horse_legs : ℕ) 
  (h1 : total_animals = 11)
  (h2 : total_legs = 30)
  (h3 : duck_legs = 2)
  (h4 : horse_legs = 4) :
  ∃ (ducks horses : ℕ),
    ducks + horses = total_animals ∧
    ducks * duck_legs + horses * horse_legs = total_legs ∧
    ducks = 7 :=
by sorry

end duck_count_l2339_233990


namespace trig_identity_l2339_233971

theorem trig_identity (a : ℝ) (h : Real.sin (π / 3 - a) = 1 / 3) :
  Real.cos (5 * π / 6 - a) = -1 / 3 := by
  sorry

end trig_identity_l2339_233971


namespace soda_packing_l2339_233916

theorem soda_packing (total : ℕ) (regular : ℕ) (diet : ℕ) (pack_size : ℕ) :
  total = 200 →
  regular = 55 →
  diet = 40 →
  pack_size = 3 →
  let energy := total - regular - diet
  let complete_packs := energy / pack_size
  let leftover := energy % pack_size
  complete_packs = 35 ∧ leftover = 0 := by
  sorry

end soda_packing_l2339_233916


namespace jill_jack_distance_difference_l2339_233925

/-- The side length of the inner square (Jack's path) in feet -/
def inner_side_length : ℕ := 300

/-- The width of the street in feet -/
def street_width : ℕ := 15

/-- The side length of the outer square (Jill's path) in feet -/
def outer_side_length : ℕ := inner_side_length + 2 * street_width

/-- The difference in distance run by Jill and Jack -/
def distance_difference : ℕ := 4 * outer_side_length - 4 * inner_side_length

theorem jill_jack_distance_difference : distance_difference = 120 := by
  sorry

end jill_jack_distance_difference_l2339_233925


namespace geometric_sequence_sum_inequality_l2339_233923

/-- Given a geometric sequence with positive terms and common ratio q > 0, q ≠ 1,
    the sum of the first and fourth terms is greater than the sum of the second and third terms. -/
theorem geometric_sequence_sum_inequality {a : ℕ → ℝ} {q : ℝ} 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_pos : ∀ n, a n > 0)
  (h_q_pos : q > 0)
  (h_q_neq_1 : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 := by
sorry

end geometric_sequence_sum_inequality_l2339_233923


namespace jamie_rice_purchase_l2339_233989

/-- The price of rice in cents per pound -/
def rice_price : ℚ := 60

/-- The price of flour in cents per pound -/
def flour_price : ℚ := 30

/-- The total amount of rice and flour bought in pounds -/
def total_amount : ℚ := 30

/-- The total amount spent in cents -/
def total_spent : ℚ := 1500

/-- The amount of rice bought in pounds -/
def rice_amount : ℚ := 20

theorem jamie_rice_purchase :
  ∃ (flour_amount : ℚ),
    rice_amount + flour_amount = total_amount ∧
    rice_price * rice_amount + flour_price * flour_amount = total_spent :=
by sorry

end jamie_rice_purchase_l2339_233989


namespace license_plate_difference_l2339_233963

/-- The number of letters in the alphabet -/
def numLetters : Nat := 26

/-- The number of digits available -/
def numDigits : Nat := 10

/-- The number of license plates Sunland can issue -/
def sunlandPlates : Nat := numLetters^5 * numDigits^2

/-- The number of license plates Moonland can issue -/
def moonlandPlates : Nat := numLetters^3 * numDigits^3

/-- The difference in the number of license plates between Sunland and Moonland -/
def plateDifference : Nat := sunlandPlates - moonlandPlates

theorem license_plate_difference : plateDifference = 1170561600 := by
  sorry

end license_plate_difference_l2339_233963


namespace roots_of_equation_l2339_233902

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 4*x + 3)*(x - 5)*(x + 1)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 3 ∨ x = 5 ∨ x = -1 := by
  sorry

end roots_of_equation_l2339_233902


namespace extremum_and_derivative_not_equivalent_l2339_233945

-- Define a function type that represents real-valued functions of a real variable
def RealFunction := ℝ → ℝ

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : RealFunction) (a : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - a| < ε → f a ≤ f x ∨ f a ≥ f x

-- Define the derivative of a function at a point
noncomputable def has_derivative_at (f : RealFunction) (a : ℝ) (f' : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a - f' * (x - a)| ≤ ε * |x - a|

-- Theorem statement
theorem extremum_and_derivative_not_equivalent :
  ∃ (f : RealFunction) (a : ℝ),
    (has_extremum f a ∧ ¬(has_derivative_at f a 0)) ∧
    ∃ (g : RealFunction) (b : ℝ),
      (has_derivative_at g b 0 ∧ ¬(has_extremum g b)) :=
sorry

end extremum_and_derivative_not_equivalent_l2339_233945


namespace necklaces_made_l2339_233912

def total_beads : ℕ := 52
def beads_per_necklace : ℕ := 2

theorem necklaces_made : total_beads / beads_per_necklace = 26 := by
  sorry

end necklaces_made_l2339_233912


namespace mirror_side_length_l2339_233951

/-- Proves that the length of each side of a square mirror is 18 inches, given the specified conditions --/
theorem mirror_side_length :
  ∀ (wall_width wall_length mirror_area : ℝ),
    wall_width = 32 →
    wall_length = 20.25 →
    mirror_area = (wall_width * wall_length) / 2 →
    ∃ (mirror_side : ℝ),
      mirror_side * mirror_side = mirror_area ∧
      mirror_side = 18 :=
by sorry

end mirror_side_length_l2339_233951


namespace rotation_90_ccw_parabola_l2339_233985

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the original function
def original_function (x : ℝ) : ℝ := x^2

-- Define the rotation operation
def rotate_90_ccw (p : Point) : Point :=
  { x := -p.y, y := p.x }

-- Define the rotated function
def rotated_function (y : ℝ) : ℝ := -y^2

-- Theorem statement
theorem rotation_90_ccw_parabola :
  ∀ (p : Point), p.y = original_function p.x →
  (rotate_90_ccw p).y = rotated_function (rotate_90_ccw p).x :=
sorry

end rotation_90_ccw_parabola_l2339_233985


namespace unique_three_digit_number_l2339_233981

theorem unique_three_digit_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ 
  (300 + n / 10) = 3 * n + 1 := by
  sorry

end unique_three_digit_number_l2339_233981


namespace fruit_basket_ratio_l2339_233965

/-- The number of bananas in the blue basket -/
def blue_bananas : ℕ := 12

/-- The number of apples in the blue basket -/
def blue_apples : ℕ := 4

/-- The number of fruits in the red basket -/
def red_fruits : ℕ := 8

/-- The total number of fruits in the blue basket -/
def blue_total : ℕ := blue_bananas + blue_apples

/-- The ratio of fruits in the red basket to the blue basket -/
def fruit_ratio : ℚ := red_fruits / blue_total

theorem fruit_basket_ratio : fruit_ratio = 1 / 2 := by
  sorry

end fruit_basket_ratio_l2339_233965


namespace fourth_intersection_point_l2339_233973

/-- Given a hyperbola and a circle with specific intersection points, 
    prove that the fourth intersection point has specific coordinates. -/
theorem fourth_intersection_point 
  (C : Set (ℝ × ℝ)) -- The circle
  (h : C.Nonempty) -- The circle is not empty
  (hyp : Set (ℝ × ℝ)) -- The hyperbola
  (hyp_eq : ∀ p ∈ hyp, p.1 * p.2 = 2) -- Equation of the hyperbola
  (intersect : C ∩ hyp = {(4, 1/2), (-2, -1), (2/3, 3), (-1/2, -4)}) -- Intersection points
  : (-1/2, -4) ∈ C ∩ hyp := by
  sorry


end fourth_intersection_point_l2339_233973


namespace prove_average_growth_rate_l2339_233969

-- Define the initial number of books borrowed in 2020
def initial_books : ℝ := 7500

-- Define the final number of books borrowed in 2022
def final_books : ℝ := 10800

-- Define the number of years between 2020 and 2022
def years : ℕ := 2

-- Define the average annual growth rate
def average_growth_rate : ℝ := 0.2

-- Theorem statement
theorem prove_average_growth_rate :
  initial_books * (1 + average_growth_rate) ^ years = final_books := by
  sorry

end prove_average_growth_rate_l2339_233969


namespace a_greater_than_b_l2339_233986

theorem a_greater_than_b : ∀ x : ℝ, (x - 3)^2 > (x - 2) * (x - 4) := by
  sorry

end a_greater_than_b_l2339_233986


namespace problem1_l2339_233967

theorem problem1 : |-3| - Real.sqrt 12 + 2 * Real.sin (30 * π / 180) + (-1) ^ 2021 = 3 - 2 * Real.sqrt 3 := by
  sorry

end problem1_l2339_233967


namespace simplify_expression_l2339_233944

theorem simplify_expression (x : ℝ) (h : x ≥ 2) :
  |2 - x| + (Real.sqrt (x - 2))^2 - Real.sqrt (4 * x^2 - 4 * x + 1) = -3 := by
  sorry

end simplify_expression_l2339_233944


namespace complex_quadrant_implies_m_range_l2339_233996

def z (m : ℝ) : ℂ := Complex.mk (m + 1) (3 - m)

def in_second_or_fourth_quadrant (z : ℂ) : Prop :=
  z.re * z.im > 0

theorem complex_quadrant_implies_m_range (m : ℝ) :
  in_second_or_fourth_quadrant (z m) → m ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by
  sorry

end complex_quadrant_implies_m_range_l2339_233996


namespace fraction_simplification_l2339_233968

theorem fraction_simplification : (1625^2 - 1618^2) / (1632^2 - 1611^2) = 1/3 := by
  sorry

end fraction_simplification_l2339_233968
