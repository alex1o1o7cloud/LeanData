import Mathlib

namespace night_day_crew_loading_ratio_l3153_315305

theorem night_day_crew_loading_ratio 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (total_boxes : ℝ) 
  (h1 : night_crew = (4 : ℝ) / 9 * day_crew) 
  (h2 : (3 : ℝ) / 4 * total_boxes = day_crew_boxes)
  (h3 : day_crew_boxes + night_crew_boxes = total_boxes) : 
  (night_crew_boxes / night_crew) / (day_crew_boxes / day_crew) = (3 : ℝ) / 4 :=
by sorry

end night_day_crew_loading_ratio_l3153_315305


namespace equivalent_annual_rate_l3153_315348

/-- Given an annual interest rate of 8% compounded quarterly, 
    the equivalent constant annual compounding rate is approximately 8.24% -/
theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) (r : ℝ) : 
  quarterly_rate = 0.08 / 4 →
  annual_rate = 0.08 →
  (1 + quarterly_rate)^4 = 1 + r →
  ∀ ε > 0, |r - 0.0824| < ε :=
sorry

end equivalent_annual_rate_l3153_315348


namespace marble_sum_l3153_315337

def marble_problem (fabian kyle miles : ℕ) : Prop :=
  fabian = 15 ∧ fabian = 3 * kyle ∧ fabian = 5 * miles

theorem marble_sum (fabian kyle miles : ℕ) 
  (h : marble_problem fabian kyle miles) : kyle + miles = 8 := by
  sorry

end marble_sum_l3153_315337


namespace abcd_sum_absolute_l3153_315310

theorem abcd_sum_absolute (a b c d : ℤ) 
  (h1 : a * b * c * d = 25)
  (h2 : a > b ∧ b > c ∧ c > d) : 
  |a + b| + |c + d| = 12 := by
sorry

end abcd_sum_absolute_l3153_315310


namespace cubic_inequality_solution_l3153_315382

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ x < -1 ∨ x > 16 := by sorry

end cubic_inequality_solution_l3153_315382


namespace not_all_face_sums_distinct_not_all_face_sums_distinct_l3153_315378

-- Define a cube type
structure Cube where
  vertices : Fin 8 → ℤ
  vertex_values : ∀ v, vertices v = 0 ∨ vertices v = 1

-- Define a function to get the sum of a face
def face_sum (c : Cube) (face : Fin 6) : ℤ :=
  sorry

-- Theorem statement
theorem not_all_face_sums_distinct (c : Cube) :
  ¬ (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → face_sum c f₁ ≠ face_sum c f₂) :=
sorry

-- For part b, we can define a similar structure and theorem
structure Cube' where
  vertices : Fin 8 → ℤ
  vertex_values : ∀ v, vertices v = 1 ∨ vertices v = -1

def face_sum' (c : Cube') (face : Fin 6) : ℤ :=
  sorry

theorem not_all_face_sums_distinct' (c : Cube') :
  ¬ (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → face_sum' c f₁ ≠ face_sum' c f₂) :=
sorry

end not_all_face_sums_distinct_not_all_face_sums_distinct_l3153_315378


namespace binomial_15_12_l3153_315362

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end binomial_15_12_l3153_315362


namespace midpoint_distance_after_move_l3153_315311

/-- Given two points A(a,b) and B(c,d) on a Cartesian plane with midpoint M(m,n),
    prove that after moving A 3 units right and 5 units up, and B 5 units left and 3 units down,
    the distance between M and the new midpoint M' is √2. -/
theorem midpoint_distance_after_move (a b c d m n : ℝ) :
  m = (a + c) / 2 →
  n = (b + d) / 2 →
  let m' := (a + 3 + c - 5) / 2
  let n' := (b + 5 + d - 3) / 2
  Real.sqrt ((m' - m)^2 + (n' - n)^2) = Real.sqrt 2 := by
  sorry

end midpoint_distance_after_move_l3153_315311


namespace calculation_proof_l3153_315335

theorem calculation_proof :
  ((-1 - (1 + 0.5) * (1/3) + (-4)) = -11/2) ∧
  ((-8^2 + 3 * (-2)^2 + (-6) + (-1/3)^2) = -521/9) := by
  sorry

end calculation_proof_l3153_315335


namespace burger_cost_theorem_l3153_315390

/-- The cost of a single burger given the total spent, total burgers, double burger cost, and number of double burgers --/
def single_burger_cost (total_spent : ℚ) (total_burgers : ℕ) (double_burger_cost : ℚ) (double_burgers : ℕ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let double_burgers_cost := double_burger_cost * double_burgers
  let single_burgers_total_cost := total_spent - double_burgers_cost
  single_burgers_total_cost / single_burgers

theorem burger_cost_theorem :
  single_burger_cost 68.50 50 1.50 37 = 1.00 := by
  sorry

end burger_cost_theorem_l3153_315390


namespace school_play_boys_count_school_play_problem_l3153_315302

/-- Given a school play with girls and boys, prove the number of boys. -/
theorem school_play_boys_count (girls : ℕ) (total_parents : ℕ) : ℕ :=
  let boys := (total_parents - 2 * girls) / 2
  by
    -- Proof goes here
    sorry

/-- The actual problem statement -/
theorem school_play_problem : school_play_boys_count 6 28 = 8 := by
  -- Proof goes here
  sorry

end school_play_boys_count_school_play_problem_l3153_315302


namespace cupcakes_per_box_l3153_315332

theorem cupcakes_per_box 
  (total_baked : ℕ) 
  (left_at_home : ℕ) 
  (boxes_given : ℕ) 
  (h1 : total_baked = 53) 
  (h2 : left_at_home = 2) 
  (h3 : boxes_given = 17) :
  (total_baked - left_at_home) / boxes_given = 3 := by
sorry

end cupcakes_per_box_l3153_315332


namespace count_special_numbers_is_126_l3153_315354

/-- A function that counts 4-digit numbers starting with 1 and having exactly two identical digits, excluding 1 and 5 as the identical digits -/
def count_special_numbers : ℕ :=
  let digits := {2, 3, 4, 6, 7, 8, 9}
  let patterns := 3  -- representing 1xxy, 1xyx, 1yxx
  let choices_for_x := Finset.card digits
  let choices_for_y := 9 - 3  -- total digits minus 1, 5, and x
  patterns * choices_for_x * choices_for_y

/-- The count of special numbers is 126 -/
theorem count_special_numbers_is_126 : count_special_numbers = 126 := by
  sorry

#eval count_special_numbers  -- This line is optional, for verification purposes

end count_special_numbers_is_126_l3153_315354


namespace correct_calculation_l3153_315384

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end correct_calculation_l3153_315384


namespace probability_four_ones_twelve_dice_l3153_315327

theorem probability_four_ones_twelve_dice :
  let n : ℕ := 12  -- total number of dice
  let k : ℕ := 4   -- number of dice showing 1
  let p : ℚ := 1/6 -- probability of rolling a 1 on a single die
  
  let probability := (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))
  
  probability = 495 * (5^8 : ℚ) / (6^12 : ℚ) := by
  sorry

end probability_four_ones_twelve_dice_l3153_315327


namespace y_coordinate_relationship_l3153_315347

/-- The quadratic function f(x) = -(x-3)^2 - 4 -/
def f (x : ℝ) : ℝ := -(x - 3)^2 - 4

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_coordinate_relationship :
  let y₁ := f (-1/2)
  let y₂ := f 1
  let y₃ := f 4
  y₁ < y₂ ∧ y₂ < y₃ := by sorry

end y_coordinate_relationship_l3153_315347


namespace tan_half_alpha_eq_two_implies_ratio_l3153_315376

theorem tan_half_alpha_eq_two_implies_ratio (α : Real) 
  (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end tan_half_alpha_eq_two_implies_ratio_l3153_315376


namespace minimum_value_expression_minimum_value_attained_l3153_315342

theorem minimum_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) ≥ 4 :=
by sorry

theorem minimum_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ p q r : ℝ, p > 0 ∧ q > 0 ∧ r > 0 ∧
    (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) < 4 + ε :=
by sorry

end minimum_value_expression_minimum_value_attained_l3153_315342


namespace all_points_above_x_axis_l3153_315341

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if a point is above or on the x-axis -/
def isAboveOrOnXAxis (p : Point) : Prop :=
  p.y ≥ 0

/-- Checks if a point is inside or on the boundary of a parallelogram -/
def isInsideOrOnParallelogram (para : Parallelogram) (p : Point) : Prop :=
  sorry  -- Definition of this function is omitted for brevity

/-- The main theorem to be proved -/
theorem all_points_above_x_axis (para : Parallelogram) 
    (h1 : para.P = ⟨-4, 4⟩) 
    (h2 : para.Q = ⟨4, 2⟩)
    (h3 : para.R = ⟨2, -2⟩)
    (h4 : para.S = ⟨-6, -4⟩) :
    ∀ p : Point, isInsideOrOnParallelogram para p → isAboveOrOnXAxis p :=
  sorry

#check all_points_above_x_axis

end all_points_above_x_axis_l3153_315341


namespace triangle_side_length_l3153_315363

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 2 →
  b + c = 7 →
  Real.cos B = -(1/4 : ℝ) →
  b = 4 := by
sorry

end triangle_side_length_l3153_315363


namespace range_of_difference_l3153_315366

theorem range_of_difference (x y : ℝ) (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) :
  27 < x - y ∧ x - y < 56 := by
  sorry

end range_of_difference_l3153_315366


namespace restaurant_expenditure_l3153_315395

theorem restaurant_expenditure (total_people : Nat) (standard_spenders : Nat) (standard_amount : ℝ) (total_spent : ℝ) :
  total_people = 8 →
  standard_spenders = 7 →
  standard_amount = 10 →
  total_spent = 88 →
  (total_spent - (standard_spenders * standard_amount)) - (total_spent / total_people) = 7 := by
  sorry

end restaurant_expenditure_l3153_315395


namespace lucy_share_l3153_315344

/-- Proves that Lucy's share is $2000 given the conditions of the problem -/
theorem lucy_share (total : ℝ) (natalie_fraction : ℝ) (rick_fraction : ℝ) 
  (h_total : total = 10000)
  (h_natalie : natalie_fraction = 1/2)
  (h_rick : rick_fraction = 3/5) : 
  total * (1 - natalie_fraction) * (1 - rick_fraction) = 2000 := by
  sorry

end lucy_share_l3153_315344


namespace probability_complement_event_correct_l3153_315360

/-- The probability of event $\overline{A}$ occurring exactly $k$ times in $n$ trials, 
    given that the probability of event $A$ occurring in each trial is $P$. -/
def probability_complement_event (n k : ℕ) (P : ℝ) : ℝ :=
  (n.choose k) * (1 - P)^k * P^(n - k)

/-- Theorem stating that the probability of event $\overline{A}$ occurring exactly $k$ times 
    in $n$ trials, given that the probability of event $A$ occurring in each trial is $P$, 
    is equal to $C_n^k(1-P)^k P^{n-k}$. -/
theorem probability_complement_event_correct (n k : ℕ) (P : ℝ) 
    (h1 : 0 ≤ P) (h2 : P ≤ 1) (h3 : k ≤ n) : 
  probability_complement_event n k P = (n.choose k) * (1 - P)^k * P^(n - k) := by
  sorry

end probability_complement_event_correct_l3153_315360


namespace k_range_l3153_315350

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x else -x^2 - 1

-- State the theorem
theorem k_range (k : ℝ) :
  (∀ x, f x ≤ k * x) → 1 ≤ k ∧ k ≤ 2 :=
by sorry

end k_range_l3153_315350


namespace range_of_m_l3153_315304

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (B m ≠ ∅) →
  (A ∪ B m = A) →
  (2 < m ∧ m ≤ 4) :=
by sorry

end range_of_m_l3153_315304


namespace set_distributive_laws_l3153_315306

theorem set_distributive_laws {α : Type*} (A B C : Set α) :
  (A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end set_distributive_laws_l3153_315306


namespace f_is_even_and_increasing_l3153_315391

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l3153_315391


namespace price_per_diaper_l3153_315375

def boxes : ℕ := 30
def packs_per_box : ℕ := 40
def diapers_per_pack : ℕ := 160
def total_revenue : ℕ := 960000

def total_diapers : ℕ := boxes * packs_per_box * diapers_per_pack

theorem price_per_diaper :
  total_revenue / total_diapers = 5 :=
by sorry

end price_per_diaper_l3153_315375


namespace shirt_price_calculation_l3153_315355

theorem shirt_price_calculation (P : ℝ) : 
  (P * (1 - 0.33333) * (1 - 0.25) * (1 - 0.2) = 15) → P = 37.50 := by
  sorry

end shirt_price_calculation_l3153_315355


namespace min_distance_from_start_l3153_315338

/-- Represents a robot's movement on a 2D plane. -/
structure RobotMovement where
  /-- The distance the robot moves per minute. -/
  speed : ℝ
  /-- The total number of minutes the robot moves. -/
  total_time : ℕ
  /-- The number of minutes before the robot starts turning. -/
  initial_straight_time : ℕ

/-- Theorem stating the minimum distance from the starting point after the robot's movement. -/
theorem min_distance_from_start (r : RobotMovement) 
  (h1 : r.speed = 10)
  (h2 : r.total_time = 9)
  (h3 : r.initial_straight_time = 1) :
  ∃ (d : ℝ), d = 10 ∧ ∀ (final_pos : ℝ × ℝ), 
    (final_pos.1^2 + final_pos.2^2).sqrt ≥ d :=
sorry

end min_distance_from_start_l3153_315338


namespace harkamal_payment_l3153_315307

/-- Calculate the total amount paid for fruits given the quantities and rates -/
def totalAmountPaid (grapeQuantity mangoQuantity grapeRate mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Theorem: Harkamal paid 1055 to the shopkeeper -/
theorem harkamal_payment : totalAmountPaid 8 9 70 55 = 1055 := by
  sorry

end harkamal_payment_l3153_315307


namespace ratatouille_cost_per_quart_l3153_315328

/-- Calculates the cost per quart of ratatouille given the ingredients and their prices. -/
theorem ratatouille_cost_per_quart 
  (eggplant_zucchini_weight : ℝ)
  (eggplant_zucchini_price : ℝ)
  (tomato_weight : ℝ)
  (tomato_price : ℝ)
  (onion_weight : ℝ)
  (onion_price : ℝ)
  (basil_weight : ℝ)
  (basil_half_pound_price : ℝ)
  (total_quarts : ℝ)
  (h1 : eggplant_zucchini_weight = 9)
  (h2 : eggplant_zucchini_price = 2)
  (h3 : tomato_weight = 4)
  (h4 : tomato_price = 3.5)
  (h5 : onion_weight = 3)
  (h6 : onion_price = 1)
  (h7 : basil_weight = 1)
  (h8 : basil_half_pound_price = 2.5)
  (h9 : total_quarts = 4) :
  (eggplant_zucchini_weight * eggplant_zucchini_price + 
   tomato_weight * tomato_price + 
   onion_weight * onion_price + 
   basil_weight * basil_half_pound_price * 2) / total_quarts = 10 :=
by sorry

end ratatouille_cost_per_quart_l3153_315328


namespace teacup_lid_arrangement_l3153_315353

def teacups : ℕ := 6
def lids : ℕ := 6
def matching_lids : ℕ := 2

theorem teacup_lid_arrangement :
  (teacups.choose matching_lids) * 
  ((teacups - matching_lids - 1) * (lids - matching_lids - 1)) = 135 := by
sorry

end teacup_lid_arrangement_l3153_315353


namespace remaining_balance_proof_l3153_315361

def gift_card_balance : ℝ := 100

def latte_price : ℝ := 3.75
def croissant_price : ℝ := 3.50
def bagel_price : ℝ := 2.25
def muffin_price : ℝ := 2.50
def special_drink_price : ℝ := 4.50
def cookie_price : ℝ := 1.25

def saturday_discount : ℝ := 0.10
def sunday_discount : ℝ := 0.20

def monday_expense : ℝ := latte_price + croissant_price + bagel_price
def tuesday_expense : ℝ := latte_price + croissant_price + muffin_price
def wednesday_expense : ℝ := latte_price + croissant_price + bagel_price
def thursday_expense : ℝ := latte_price + croissant_price + muffin_price
def friday_expense : ℝ := special_drink_price + croissant_price + bagel_price
def saturday_expense : ℝ := latte_price + croissant_price * (1 - saturday_discount)
def sunday_expense : ℝ := latte_price * (1 - sunday_discount) + croissant_price

def cookie_expense : ℝ := 5 * cookie_price

def total_expense : ℝ := monday_expense + tuesday_expense + wednesday_expense + thursday_expense + 
                         friday_expense + saturday_expense + sunday_expense + cookie_expense

theorem remaining_balance_proof : 
  gift_card_balance - total_expense = 31.60 := by
  sorry

end remaining_balance_proof_l3153_315361


namespace sqrt_comparison_l3153_315345

theorem sqrt_comparison : Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end sqrt_comparison_l3153_315345


namespace point_m_locations_l3153_315365

/-- Given a line segment AC with point B on AC such that AB = 2 and BC = 1,
    prove that the only points M on the line AC that satisfy AM + MB = CM
    are at x = 1 and x = -1, where A is at x = 0 and C is at x = 3. -/
theorem point_m_locations (A B C M : ℝ) (h1 : 0 < B) (h2 : B < 3) (h3 : B = 2) :
  (M < 0 ∨ 0 ≤ M ∧ M ≤ 3) →
  (abs (M - 0) + abs (M - 2) = abs (M - 3)) ↔ (M = 1 ∨ M = -1) :=
by sorry

end point_m_locations_l3153_315365


namespace f_properties_l3153_315340

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem f_properties :
  (∃ a b : ℝ, a = -2 ∧ b = 0 ∧ ∀ x ∈ Set.Ioo a b, StrictMonoOn f (Set.Ioo a b)) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = x₁ - 2012 ∧ f x₂ = x₂ - 2012) :=
by sorry

end f_properties_l3153_315340


namespace inequality_proof_l3153_315329

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end inequality_proof_l3153_315329


namespace triangle_height_l3153_315349

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 10 → area = 50 → area = (base * height) / 2 → height = 10 := by
  sorry

end triangle_height_l3153_315349


namespace eighteenth_term_is_three_l3153_315379

/-- An equal sum sequence with public sum 5 and a₁ = 2 -/
def EqualSumSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n + a (n + 1) = 5) ∧ a 1 = 2

/-- The 18th term of the equal sum sequence is 3 -/
theorem eighteenth_term_is_three (a : ℕ → ℕ) (h : EqualSumSequence a) : a 18 = 3 := by
  sorry

end eighteenth_term_is_three_l3153_315379


namespace sum_of_tangent_slopes_l3153_315325

/-- The parabola P with equation y = x^2 + 5x -/
def P (x y : ℝ) : Prop := y = x^2 + 5*x

/-- The point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- The equation whose roots are the slopes of lines through Q tangent to P -/
def tangent_slope_equation (m : ℝ) : Prop := m^2 - 50*m + 1 = 0

/-- The sum of the roots of the tangent slope equation is 50 -/
theorem sum_of_tangent_slopes : 
  ∃ r s : ℝ, tangent_slope_equation r ∧ tangent_slope_equation s ∧ r + s = 50 :=
sorry

end sum_of_tangent_slopes_l3153_315325


namespace limit_cube_minus_one_over_x_minus_one_l3153_315312

theorem limit_cube_minus_one_over_x_minus_one : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |(x^3 - 1) / (x - 1) - 3| < ε :=
sorry

end limit_cube_minus_one_over_x_minus_one_l3153_315312


namespace smallest_divisible_by_15_11_12_l3153_315398

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ+, (∀ m : ℕ+, m < n → ¬(15 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m)) ∧ (15 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n) := by
  sorry

end smallest_divisible_by_15_11_12_l3153_315398


namespace race_distance_l3153_315386

theorem race_distance (speed_A speed_B speed_C : ℝ) : 
  (speed_A / speed_B = 1000 / 900) →
  (speed_B / speed_C = 800 / 700) →
  (∃ D : ℝ, D > 0 ∧ D * (speed_A / speed_C - 1) = 127.5) →
  (∃ D : ℝ, D > 0 ∧ D * (speed_A / speed_C - 1) = 127.5 ∧ D = 600) :=
by sorry

end race_distance_l3153_315386


namespace problem_one_problem_two_problem_three_problem_four_l3153_315322

-- Problem 1
theorem problem_one : 27 - (-12) + 3 - 7 = 35 := by sorry

-- Problem 2
theorem problem_two : (-3 - 1/3) * 2/5 * (-2 - 1/2) / (-10/7) = -7/3 := by sorry

-- Problem 3
theorem problem_three : (3/4 - 7/8 - 7/12) * (-12) = 17/2 := by sorry

-- Problem 4
theorem problem_four : 4 / (-2/3)^2 + 1 + (-1)^2023 = 9 := by sorry

end problem_one_problem_two_problem_three_problem_four_l3153_315322


namespace max_gcd_13n_plus_4_8n_plus_3_l3153_315389

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) ∧
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) = 7) :=
by sorry

end max_gcd_13n_plus_4_8n_plus_3_l3153_315389


namespace c_months_is_eleven_l3153_315372

/-- Represents the rental scenario for a pasture -/
structure PastureRental where
  total_rent : ℕ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  b_payment : ℕ

/-- Calculates the number of months c put in horses -/
def calculate_c_months (rental : PastureRental) : ℕ :=
  sorry

/-- Theorem stating that given the rental conditions, c put in horses for 11 months -/
theorem c_months_is_eleven (rental : PastureRental) 
  (h1 : rental.total_rent = 841)
  (h2 : rental.a_horses = 12)
  (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16)
  (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18)
  (h7 : rental.b_payment = 348) :
  calculate_c_months rental = 11 :=
by sorry

end c_months_is_eleven_l3153_315372


namespace circle_symmetry_range_l3153_315357

/-- A circle with equation x^2 + y^2 - 2x + 6y + 5a = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 6*p.2 + 5*a = 0}

/-- A line with equation y = x + 2b -/
def Line (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 2*b}

/-- The circle is symmetric about the line -/
def IsSymmetric (a b : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), center ∈ Circle a ∧ center ∈ Line b

theorem circle_symmetry_range (a b : ℝ) :
  IsSymmetric a b → a - b ∈ Set.Iio 4 :=
sorry

end circle_symmetry_range_l3153_315357


namespace fraction_problem_l3153_315303

theorem fraction_problem (f n : ℚ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1/5 := by
  sorry

end fraction_problem_l3153_315303


namespace train_length_calculation_l3153_315387

-- Define the given values
def train_speed : Real := 100  -- km/h
def motorbike_speed : Real := 64  -- km/h
def overtake_time : Real := 18  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let train_speed_ms : Real := train_speed * 1000 / 3600
  let motorbike_speed_ms : Real := motorbike_speed * 1000 / 3600
  let relative_speed : Real := train_speed_ms - motorbike_speed_ms
  let train_length : Real := relative_speed * overtake_time
  train_length = 180 := by
  sorry

end train_length_calculation_l3153_315387


namespace surface_area_of_specific_solid_l3153_315316

/-- A solid formed by unit cubes -/
structure CubeSolid where
  num_cubes : ℕ
  height : ℕ
  width : ℕ

/-- The surface area of a CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ := by sorry

/-- The theorem stating the surface area of the specific solid -/
theorem surface_area_of_specific_solid :
  ∃ (solid : CubeSolid),
    solid.num_cubes = 10 ∧
    solid.height = 3 ∧
    solid.width = 4 ∧
    surface_area solid = 34 := by sorry

end surface_area_of_specific_solid_l3153_315316


namespace function_greater_than_three_sixteenths_l3153_315334

/-- The function f(x) = x^2 + 2mx + m is greater than 3/16 for all x if and only if 1/4 < m < 3/4 -/
theorem function_greater_than_three_sixteenths (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*m*x + m > 3/16) ↔ (1/4 < m ∧ m < 3/4) :=
sorry

end function_greater_than_three_sixteenths_l3153_315334


namespace expression_bounds_l3153_315358

theorem expression_bounds (x y : ℝ) (h : abs x + abs y = 13) :
  0 ≤ x^2 + 7*x - 3*y + y^2 ∧ x^2 + 7*x - 3*y + y^2 ≤ 260 :=
by sorry

end expression_bounds_l3153_315358


namespace gvidon_descendants_l3153_315364

/-- The number of sons King Gvidon had -/
def kings_sons : ℕ := 5

/-- The number of descendants who had exactly 3 sons each -/
def descendants_with_sons : ℕ := 100

/-- The number of sons each fertile descendant had -/
def sons_per_descendant : ℕ := 3

/-- The total number of descendants of King Gvidon -/
def total_descendants : ℕ := kings_sons + descendants_with_sons * sons_per_descendant

theorem gvidon_descendants :
  total_descendants = 305 :=
sorry

end gvidon_descendants_l3153_315364


namespace triplet_convergence_l3153_315330

/-- Given a triplet of numbers, compute the absolute differences -/
def absDiff (t : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := t
  (|a - b|, |b - c|, |c - a|)

/-- Generate the sequence of triplets -/
def tripletSeq (x y z : ℝ) : ℕ → ℝ × ℝ × ℝ
  | 0 => (x, y, z)
  | n + 1 => absDiff (tripletSeq x y z n)

theorem triplet_convergence (y z : ℝ) :
  (∃ n : ℕ, tripletSeq 1 y z n = (1, y, z)) → y = 1 ∧ z = 0 := by
  sorry

end triplet_convergence_l3153_315330


namespace polynomial_factor_l3153_315370

theorem polynomial_factor (a b c : ℤ) (x : ℚ) : 
  a = 1 ∧ b = -1 ∧ c = -8 →
  ∃ k : ℚ, (x^2 - 2*x - 1) * (2*a*x + k) = 2*a*x^3 + b*x^2 + c*x - 3 :=
by sorry

end polynomial_factor_l3153_315370


namespace min_value_of_function_min_value_attained_l3153_315369

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  (4 * x + 1 / (4 * x - 5)) ≥ 7 := by
  sorry

theorem min_value_attained (x : ℝ) (h : x > 5/4) :
  ∃ x₀ > 5/4, 4 * x₀ + 1 / (4 * x₀ - 5) = 7 := by
  sorry

end min_value_of_function_min_value_attained_l3153_315369


namespace smallest_nonzero_real_l3153_315381

theorem smallest_nonzero_real : ∃ (p q : ℕ+) (x : ℝ),
  x = -Real.sqrt p / q ∧
  x ≠ 0 ∧
  (∀ y : ℝ, y ≠ 0 → y⁻¹ = y - Real.sqrt (y^2) → |x| ≤ |y|) ∧
  (∀ (a : ℕ+), a^2 ∣ p → a = 1) ∧
  x⁻¹ = x - Real.sqrt (x^2) ∧
  p + q = 4 := by
sorry

end smallest_nonzero_real_l3153_315381


namespace arithmetic_sequence_ratio_l3153_315380

/-- Two arithmetic sequences and their sum ratios -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, S n = (n : ℚ) / 2 * (a 1 + a n)) →
  (∀ n : ℕ, T n = (n : ℚ) / 2 * (b 1 + b n)) →
  (∀ n : ℕ, S n / T n = (7 * n + 2 : ℚ) / (n + 3)) →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  (∀ n : ℕ, b (n + 1) - b n = b 2 - b 1) →
  a 7 / b 7 = 93 / 16 := by
  sorry

end arithmetic_sequence_ratio_l3153_315380


namespace repeating_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l3153_315394

/-- The repeating decimal 0.134134134... as a real number -/
def repeating_decimal : ℝ := 0.134134134

/-- The fraction representation of the repeating decimal -/
def fraction : ℚ := 134 / 999

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = fraction := by sorry

theorem fraction_is_lowest_terms : 
  Nat.gcd 134 999 = 1 := by sorry

theorem sum_of_numerator_and_denominator : 
  134 + 999 = 1133 := by sorry

#eval 134 + 999  -- To verify the result

end repeating_decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l3153_315394


namespace counterexample_exists_l3153_315367

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end counterexample_exists_l3153_315367


namespace quadratic_roots_sum_product_l3153_315368

theorem quadratic_roots_sum_product (p q : ℝ) :
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 10 ∧ r₁ * r₂ = 15 ∧ 
      3 * r₁^2 - p * r₁ + q = 0 ∧ 
      3 * r₂^2 - p * r₂ + q = 0)) →
  p = 30 ∧ q = 45 := by
sorry

end quadratic_roots_sum_product_l3153_315368


namespace four_digit_multiple_of_19_l3153_315315

theorem four_digit_multiple_of_19 (a : ℕ) : 
  (2000 + 100 * a + 17) % 19 = 0 → a = 7 := by
sorry

end four_digit_multiple_of_19_l3153_315315


namespace solution_set_equation_l3153_315336

theorem solution_set_equation (x : ℝ) : 
  (1 / (x^2 + 12*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 14*x - 8) = 0) ↔ 
  (x = 2 ∨ x = -4 ∨ x = 1 ∨ x = -8) :=
by sorry

end solution_set_equation_l3153_315336


namespace sum_of_solutions_eq_18_l3153_315300

theorem sum_of_solutions_eq_18 : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, x^2 - 8*x + 21 = |x - 5| + 4) ∧ 
  (∀ x : ℝ, x^2 - 8*x + 21 = |x - 5| + 4 → x ∈ S) ∧
  (S.sum id = 18) :=
sorry

end sum_of_solutions_eq_18_l3153_315300


namespace floor_properties_l3153_315385

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_properties (x y : ℝ) :
  (x - 1 < floor x) ∧
  (floor x - floor y - 1 < x - y) ∧
  (x - y < floor x - floor y + 1) ∧
  (x^2 + 1/3 > floor x) :=
by sorry

end floor_properties_l3153_315385


namespace total_evening_sales_l3153_315339

/-- Calculates the total evening sales given the conditions of the problem -/
theorem total_evening_sales :
  let remy_bottles : ℕ := 55
  let nick_bottles : ℕ := remy_bottles - 6
  let price_per_bottle : ℚ := 1/2
  let morning_sales : ℚ := (remy_bottles + nick_bottles : ℚ) * price_per_bottle
  let evening_sales : ℚ := morning_sales + 3
  evening_sales = 55 := by sorry

end total_evening_sales_l3153_315339


namespace prob_four_to_five_l3153_315388

-- Define the possible on-times
inductive OnTime
  | Seven
  | SevenThirty
  | Eight
  | EightThirty
  | Nine

-- Define the probability space
def Ω : Type := OnTime × ℝ

-- Define the probability measure
axiom P : Set Ω → ℝ

-- Define the uniform distribution of on-times
axiom uniform_on_time : ∀ t : OnTime, P {ω : Ω | ω.1 = t} = 1/5

-- Define the uniform distribution of off-times
axiom uniform_off_time : ∀ a b : ℝ, 
  23 ≤ a ∧ a < b ∧ b ≤ 25 → P {ω : Ω | a ≤ ω.2 ∧ ω.2 ≤ b} = (b - a) / 2

-- Define the event where 4 < t < 5
def E : Set Ω :=
  {ω : Ω | 
    (ω.1 = OnTime.Seven ∧ 23 < ω.2 ∧ ω.2 < 24) ∨
    (ω.1 = OnTime.SevenThirty ∧ 23.5 < ω.2 ∧ ω.2 < 24.5) ∨
    (ω.1 = OnTime.Eight ∧ 24 < ω.2 ∧ ω.2 < 25) ∨
    (ω.1 = OnTime.EightThirty ∧ 24.5 < ω.2 ∧ ω.2 ≤ 25)}

-- Theorem to prove
theorem prob_four_to_five : P E = 7/20 := by
  sorry

end prob_four_to_five_l3153_315388


namespace largest_multiple_of_11_under_100_l3153_315317

theorem largest_multiple_of_11_under_100 : ∃ n : ℕ, n * 11 = 99 ∧ 
  (∀ m : ℕ, m * 11 < 100 → m * 11 ≤ 99) := by
  sorry

end largest_multiple_of_11_under_100_l3153_315317


namespace violet_distance_in_race_l3153_315343

/-- The distance Violet has covered in a race -/
def violet_distance (race_length : ℕ) (aubrey_finish : ℕ) (violet_remaining : ℕ) : ℕ :=
  aubrey_finish - violet_remaining

/-- Theorem: In a 1 km race, if Aubrey finishes when Violet is 279 meters from the finish line,
    then Violet has covered 721 meters -/
theorem violet_distance_in_race : 
  violet_distance 1000 1000 279 = 721 := by
  sorry

end violet_distance_in_race_l3153_315343


namespace simplify_expression_l3153_315313

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x*(x - 4) = 2*x^2 + 4 := by
  sorry

end simplify_expression_l3153_315313


namespace max_intersections_quadrilateral_hexagon_l3153_315359

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The maximum number of intersection points between the boundaries of a quadrilateral and a hexagon -/
def max_intersection_points : ℕ := quadrilateral_sides * hexagon_sides

/-- Theorem stating that the maximum number of intersection points between 
    the boundaries of a quadrilateral and a hexagon is 24 -/
theorem max_intersections_quadrilateral_hexagon : 
  max_intersection_points = 24 := by sorry

end max_intersections_quadrilateral_hexagon_l3153_315359


namespace pizza_toppings_combinations_l3153_315392

/-- The number of combinations of k items chosen from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of available toppings -/
def num_toppings : ℕ := 7

/-- The number of toppings to choose -/
def toppings_to_choose : ℕ := 3

theorem pizza_toppings_combinations :
  binomial num_toppings toppings_to_choose = 35 := by
  sorry

end pizza_toppings_combinations_l3153_315392


namespace recliner_price_drop_l3153_315396

/-- Proves that a 80% increase in sales and a 44% increase in gross revenue
    results in a 20% price drop -/
theorem recliner_price_drop (P N : ℝ) (P' N' : ℝ) :
  N' = 1.8 * N →
  P' * N' = 1.44 * (P * N) →
  P' = 0.8 * P :=
by sorry

end recliner_price_drop_l3153_315396


namespace m_salary_percentage_l3153_315323

def total_salary : ℝ := 550
def n_salary : ℝ := 250

theorem m_salary_percentage : 
  (total_salary - n_salary) / n_salary * 100 = 120 := by
  sorry

end m_salary_percentage_l3153_315323


namespace total_shaded_area_is_75_over_4_l3153_315351

/-- Represents a truncated square-based pyramid -/
structure TruncatedPyramid where
  base_side : ℝ
  top_side : ℝ
  height : ℝ

/-- Calculate the total shaded area of the truncated pyramid -/
def total_shaded_area (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The main theorem stating that the total shaded area is 75/4 -/
theorem total_shaded_area_is_75_over_4 :
  ∀ (p : TruncatedPyramid),
  p.base_side = 7 ∧ p.top_side = 1 ∧ p.height = 4 →
  total_shaded_area p = 75 / 4 := by
  sorry

end total_shaded_area_is_75_over_4_l3153_315351


namespace rachel_colored_pictures_l3153_315321

def coloring_book_problem (book1_pictures book2_pictures remaining_pictures : ℕ) : Prop :=
  let total_pictures := book1_pictures + book2_pictures
  let colored_pictures := total_pictures - remaining_pictures
  colored_pictures = 44

theorem rachel_colored_pictures :
  coloring_book_problem 23 32 11 := by
  sorry

end rachel_colored_pictures_l3153_315321


namespace triangle_angle_measure_l3153_315314

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = √2 and b = 2 sin B + cos B = √2, then angle A measures π/6 radians. -/
theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  (a = Real.sqrt 2) →
  (b = 2 * Real.sin B + Real.cos B) →
  (b = Real.sqrt 2) →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A = π / 6) := by
  sorry

end triangle_angle_measure_l3153_315314


namespace general_term_equals_closed_form_l3153_315309

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := (2 * n - 1 : ℚ) + n / (2 * n + 1 : ℚ)

/-- The proposed closed form of the general term -/
def a_closed (n : ℕ) : ℚ := (4 * n^2 + n - 1 : ℚ) / (2 * n + 1 : ℚ)

/-- Theorem stating that the general term equals the closed form -/
theorem general_term_equals_closed_form (n : ℕ) : a n = a_closed n := by
  sorry

end general_term_equals_closed_form_l3153_315309


namespace cosine_rational_values_l3153_315318

theorem cosine_rational_values (α : ℚ) (h : ∃ (q : ℚ), q = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) = 0 ∨ 
  Real.cos (α * Real.pi) = (1/2 : ℝ) ∨ 
  Real.cos (α * Real.pi) = -(1/2 : ℝ) ∨ 
  Real.cos (α * Real.pi) = 1 ∨ 
  Real.cos (α * Real.pi) = -1 := by
sorry

end cosine_rational_values_l3153_315318


namespace green_hats_count_l3153_315333

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_cost = 6)
  (h3 : green_cost = 7)
  (h4 : total_cost = 548) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 38 := by
sorry

end green_hats_count_l3153_315333


namespace absolute_value_equation_solution_l3153_315346

theorem absolute_value_equation_solution :
  ∀ y : ℝ, (|y - 4| + 3 * y = 11) ↔ (y = 15/4 ∨ y = 7/2) := by sorry

end absolute_value_equation_solution_l3153_315346


namespace nth_inequality_l3153_315356

theorem nth_inequality (x : ℝ) (n : ℕ) (h : x > 0) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
  sorry

end nth_inequality_l3153_315356


namespace problem_statement_l3153_315393

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (∃ min_val : ℝ, min_val = 144/49 ∧ 
    ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
      (x + 1)^2 + 4*y^2 + 9*z^2 ≥ min_val) ∧
  (1 / (Real.sqrt a + Real.sqrt b) + 
   1 / (Real.sqrt b + Real.sqrt c) + 
   1 / (Real.sqrt c + Real.sqrt a) ≥ 3 * Real.sqrt 3 / 2) :=
by sorry

end problem_statement_l3153_315393


namespace stating_max_coins_three_weighings_l3153_315324

/-- Represents the number of weighings available. -/
def num_weighings : ℕ := 3

/-- Represents the number of possible outcomes for each weighing. -/
def outcomes_per_weighing : ℕ := 3

/-- Calculates the total number of possible outcomes for all weighings. -/
def total_outcomes : ℕ := outcomes_per_weighing ^ num_weighings

/-- Represents the maximum number of coins that can be determined. -/
def max_coins : ℕ := 12

/-- 
Theorem stating that the maximum number of coins that can be determined
with three weighings, identifying both the counterfeit coin and whether
it's lighter or heavier, is 12.
-/
theorem max_coins_three_weighings :
  (2 * max_coins ≤ total_outcomes) ∧
  (2 * (max_coins + 1) > total_outcomes) :=
sorry

end stating_max_coins_three_weighings_l3153_315324


namespace millet_dominant_on_wednesday_l3153_315331

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℝ
  other : ℝ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := 0.8 * state.millet + 0.3,
    other := 0.5 * state.other + 0.7 }

/-- Checks if millet constitutes more than half of the seeds -/
def milletDominant (state : FeederState) : Prop :=
  state.millet > (state.millet + state.other) / 2

/-- Initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, millet := 0.3, other := 0.7 }

/-- Theorem stating that millet becomes dominant on day 3 (Wednesday) -/
theorem millet_dominant_on_wednesday :
  let day3 := nextDay (nextDay initialState)
  milletDominant day3 ∧ ¬milletDominant (nextDay initialState) :=
sorry

end millet_dominant_on_wednesday_l3153_315331


namespace cylinder_volume_l3153_315383

/-- Represents a cylinder formed by rotating a rectangle around one of its sides. -/
structure Cylinder where
  /-- The area of the original rectangle. -/
  S : ℝ
  /-- The circumference of the circle described by the intersection point of the rectangle's diagonals. -/
  C : ℝ
  /-- Ensure that S and C are positive. -/
  S_pos : S > 0
  C_pos : C > 0

/-- The volume of the cylinder. -/
def volume (cyl : Cylinder) : ℝ := cyl.S * cyl.C

/-- Theorem stating that the volume of the cylinder is equal to the product of S and C. -/
theorem cylinder_volume (cyl : Cylinder) : volume cyl = cyl.S * cyl.C := by
  sorry

end cylinder_volume_l3153_315383


namespace smallest_square_cover_l3153_315374

/-- The side length of the smallest square that can be covered by 3x4 rectangles -/
def minSquareSide : ℕ := 12

/-- The number of 3x4 rectangles needed to cover the square -/
def numRectangles : ℕ := 12

/-- The area of a 3x4 rectangle -/
def rectangleArea : ℕ := 3 * 4

theorem smallest_square_cover :
  (minSquareSide * minSquareSide) % rectangleArea = 0 ∧
  numRectangles * rectangleArea = minSquareSide * minSquareSide ∧
  ∀ n : ℕ, n < minSquareSide → (n * n) % rectangleArea ≠ 0 :=
by sorry

#check smallest_square_cover

end smallest_square_cover_l3153_315374


namespace grinder_price_correct_l3153_315397

/-- Represents the purchase and sale of two items with given profit/loss percentages --/
structure TwoItemSale where
  grinder_price : ℝ
  mobile_price : ℝ
  grinder_loss_percent : ℝ
  mobile_profit_percent : ℝ
  total_profit : ℝ

/-- The specific scenario described in the problem --/
def problem_scenario : TwoItemSale where
  grinder_price := 15000  -- This is what we want to prove
  mobile_price := 10000
  grinder_loss_percent := 4
  mobile_profit_percent := 10
  total_profit := 400

/-- Theorem stating that the given scenario satisfies the problem conditions --/
theorem grinder_price_correct (s : TwoItemSale) : 
  s.mobile_price = 10000 ∧
  s.grinder_loss_percent = 4 ∧
  s.mobile_profit_percent = 10 ∧
  s.total_profit = 400 →
  s.grinder_price = 15000 :=
by
  sorry

#check grinder_price_correct problem_scenario

end grinder_price_correct_l3153_315397


namespace simplify_expression_l3153_315373

theorem simplify_expression (x y : ℝ) : (2*x^2 - x*y) - (x^2 + x*y - 8) = x^2 - 2*x*y + 8 := by
  sorry

end simplify_expression_l3153_315373


namespace large_kangaroo_count_toy_store_kangaroos_l3153_315371

theorem large_kangaroo_count (total : ℕ) (empty_pouch : ℕ) (small_per_pouch : ℕ) : ℕ :=
  let full_pouch := total - empty_pouch
  let small_kangaroos := full_pouch * small_per_pouch
  total - small_kangaroos

theorem toy_store_kangaroos :
  large_kangaroo_count 100 77 3 = 31 := by
  sorry

end large_kangaroo_count_toy_store_kangaroos_l3153_315371


namespace polygon_E_largest_area_l3153_315308

/-- Represents a polygon composed of unit squares and right triangles --/
structure Polygon where
  squares : ℕ
  triangles : ℕ

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℚ :=
  p.squares + p.triangles / 2

/-- Theorem stating that polygon E has the largest area --/
theorem polygon_E_largest_area (A B C D E : Polygon)
  (hA : A = ⟨5, 0⟩)
  (hB : B = ⟨5, 0⟩)
  (hC : C = ⟨5, 0⟩)
  (hD : D = ⟨4, 1⟩)
  (hE : E = ⟨5, 1⟩) :
  area E ≥ area A ∧ area E ≥ area B ∧ area E ≥ area C ∧ area E ≥ area D := by
  sorry

#check polygon_E_largest_area

end polygon_E_largest_area_l3153_315308


namespace linear_function_composition_l3153_315352

/-- A linear function from ℝ to ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 9) →
  (∀ x, f x = 2 * x + 3) ∨ (∀ x, f x = -2 * x - 9) :=
by sorry

end linear_function_composition_l3153_315352


namespace equation_solution_l3153_315319

theorem equation_solution : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / (1 / 2) ∧ x = -21 := by
  sorry

end equation_solution_l3153_315319


namespace hyperbola_line_intersection_fixed_point_l3153_315326

/-- Represents a hyperbola with center at origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Hyperbola.standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.k * x + l.m

def circle_diameter_passes_through (A B D : Point) : Prop :=
  (A.y / (A.x + D.x)) * (B.y / (B.x + D.x)) = -1

theorem hyperbola_line_intersection_fixed_point
  (h : Hyperbola)
  (l : Line)
  (A B D : Point) :
  h.a = 2 →
  h.b = 1 →
  h.e = Real.sqrt 5 / 2 →
  Hyperbola.standard_equation h A.x A.y →
  Hyperbola.standard_equation h B.x B.y →
  Line.equation l A.x A.y →
  Line.equation l B.x B.y →
  D.x = -2 →
  D.y = 0 →
  A ≠ D →
  B ≠ D →
  circle_diameter_passes_through A B D →
  ∃ P : Point, P.x = -10/3 ∧ P.y = 0 ∧ Line.equation l P.x P.y :=
sorry

end hyperbola_line_intersection_fixed_point_l3153_315326


namespace y_intercept_of_line_l3153_315320

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

/-- The line equation y = -3x + 5 -/
def line (x : ℝ) : ℝ := -3 * x + 5

theorem y_intercept_of_line :
  y_intercept line = 5 := by sorry

end y_intercept_of_line_l3153_315320


namespace people_per_institution_l3153_315399

theorem people_per_institution 
  (total_institutions : ℕ) 
  (total_people : ℕ) 
  (h1 : total_institutions = 6) 
  (h2 : total_people = 480) : 
  total_people / total_institutions = 80 := by
  sorry

end people_per_institution_l3153_315399


namespace product_of_difference_and_sum_of_squares_l3153_315377

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a^2 + b^2 = 50) : 
  a * b = 7 := by
sorry

end product_of_difference_and_sum_of_squares_l3153_315377


namespace monday_average_is_7_l3153_315301

/-- The average number of birds Kendra saw at each site on Monday -/
def monday_average : ℝ := sorry

/-- The number of sites visited on Monday -/
def monday_sites : ℕ := 5

/-- The number of sites visited on Tuesday -/
def tuesday_sites : ℕ := 5

/-- The number of sites visited on Wednesday -/
def wednesday_sites : ℕ := 10

/-- The average number of birds seen at each site on Tuesday -/
def tuesday_average : ℝ := 5

/-- The average number of birds seen at each site on Wednesday -/
def wednesday_average : ℝ := 8

/-- The overall average number of birds seen at each site across all three days -/
def overall_average : ℝ := 7

theorem monday_average_is_7 :
  monday_average = 7 :=
by sorry

end monday_average_is_7_l3153_315301
