import Mathlib

namespace pucks_not_in_original_position_l2473_247305

/-- Represents the arrangement of three pucks -/
inductive Arrangement
  | ABC
  | ACB
  | BAC
  | BCA
  | CAB
  | CBA

/-- Represents a single swap operation -/
def swap : Arrangement → Arrangement
  | Arrangement.ABC => Arrangement.BAC
  | Arrangement.ACB => Arrangement.CAB
  | Arrangement.BAC => Arrangement.ABC
  | Arrangement.BCA => Arrangement.CBA
  | Arrangement.CAB => Arrangement.ACB
  | Arrangement.CBA => Arrangement.BCA

/-- Applies n swaps to the initial arrangement -/
def applySwaps (n : Nat) (init : Arrangement) : Arrangement :=
  match n with
  | 0 => init
  | n + 1 => swap (applySwaps n init)

/-- Theorem stating that after 25 swaps, the arrangement cannot be the same as the initial one -/
theorem pucks_not_in_original_position (init : Arrangement) : 
  applySwaps 25 init ≠ init :=
sorry

end pucks_not_in_original_position_l2473_247305


namespace dividend_calculation_l2473_247367

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 36)
  (h_divisor : divisor = 85)
  (h_remainder : remainder = 26) :
  (divisor * quotient) + remainder = 3086 := by
  sorry

end dividend_calculation_l2473_247367


namespace girls_in_school_l2473_247368

/-- The number of girls in a school given stratified sampling information -/
theorem girls_in_school (total : ℕ) (sample : ℕ) (girl_sample : ℕ) 
  (h_total : total = 1600)
  (h_sample : sample = 200)
  (h_girl_sample : girl_sample = 95)
  (h_ratio : (girl_sample : ℚ) / sample = (↑girls : ℚ) / total) :
  girls = 760 :=
sorry

end girls_in_school_l2473_247368


namespace tims_photos_l2473_247338

theorem tims_photos (total : ℕ) (toms_photos : ℕ) (pauls_extra : ℕ) : 
  total = 152 → toms_photos = 38 → pauls_extra = 10 →
  ∃ (tims_photos : ℕ), 
    tims_photos + toms_photos + (tims_photos + pauls_extra) = total ∧ 
    tims_photos = 52 := by
  sorry

end tims_photos_l2473_247338


namespace triangle_theorem_l2473_247378

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  angle_sum : A + B + C = π
  tan_condition : 2 * (Real.tan A + Real.tan B) = Real.tan A / Real.cos B + Real.tan B / Real.cos A

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) : 
  t.a + t.b = 2 * t.c ∧ Real.cos t.C ≥ 1/2 ∧ ∃ (t' : Triangle), Real.cos t'.C = 1/2 := by
  sorry

end triangle_theorem_l2473_247378


namespace parallel_line_through_point_l2473_247397

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Define the point that the parallel line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem parallel_line_through_point :
  (∀ x y : ℝ, given_line x y ↔ 2 * x - 3 * y + 5 = 0) →
  parallel_line point.1 point.2 ∧
  (∀ x y : ℝ, parallel_line x y ↔ 2 * x - 3 * y + 7 = 0) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, given_line x y ↔ parallel_line x y) :=
by sorry

end parallel_line_through_point_l2473_247397


namespace star_equality_implies_x_eq_five_l2473_247361

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem star_equality_implies_x_eq_five :
  ∀ y : ℤ, star 4 5 1 1 = star x y 2 3 → x = 5 := by
  sorry

end star_equality_implies_x_eq_five_l2473_247361


namespace credit_card_more_beneficial_l2473_247394

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 8000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.0075

/-- Represents the monthly interest rate on the debit card balance -/
def debit_interest_rate : ℝ := 0.005

/-- Calculates the total income from using the credit card -/
def credit_card_income (amount : ℝ) : ℝ :=
  amount * credit_cashback_rate + amount * debit_interest_rate

/-- Calculates the total income from using the debit card -/
def debit_card_income (amount : ℝ) : ℝ :=
  amount * debit_cashback_rate

/-- Theorem stating that the credit card is more beneficial -/
theorem credit_card_more_beneficial :
  credit_card_income purchase_amount > debit_card_income purchase_amount :=
by sorry


end credit_card_more_beneficial_l2473_247394


namespace triangle_properties_l2473_247331

-- Define what it means for a triangle to be equilateral
def is_equilateral (triangle : Type) : Prop := sorry

-- Define what it means for a triangle to be isosceles
def is_isosceles (triangle : Type) : Prop := sorry

-- The main theorem
theorem triangle_properties :
  (∀ t, is_equilateral t → is_isosceles t) ∧
  (∃ t, is_isosceles t ∧ ¬is_equilateral t) ∧
  (∃ t, ¬is_equilateral t ∧ is_isosceles t) := by sorry

end triangle_properties_l2473_247331


namespace trig_identity_l2473_247340

theorem trig_identity : 
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) = 
  - (Real.sqrt 3) / 2 := by sorry

end trig_identity_l2473_247340


namespace richmond_tigers_ticket_sales_l2473_247382

/-- The number of tickets sold by the Richmond Tigers in the second half of the season -/
def second_half_tickets (total : ℕ) (first_half : ℕ) : ℕ :=
  total - first_half

/-- Theorem stating that the number of tickets sold in the second half of the season is 5703 -/
theorem richmond_tigers_ticket_sales :
  second_half_tickets 9570 3867 = 5703 := by
  sorry

end richmond_tigers_ticket_sales_l2473_247382


namespace ellipse_equation_from_conditions_l2473_247359

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The area of the quadrilateral formed by the vertices of an ellipse -/
def quadrilateral_area (E : Ellipse) : ℝ := 2 * E.a * E.b

theorem ellipse_equation_from_conditions (E : Ellipse) 
  (h_vertex : ellipse_equation E 0 (-2))
  (h_area : quadrilateral_area E = 4 * Real.sqrt 5) :
  ∀ x y, ellipse_equation E x y ↔ x^2 / 5 + y^2 / 4 = 1 := by
  sorry

#check ellipse_equation_from_conditions

end ellipse_equation_from_conditions_l2473_247359


namespace lcm_sum_ratio_problem_l2473_247311

theorem lcm_sum_ratio_problem (A B x y : ℕ+) : 
  Nat.lcm A B = 60 →
  A + B = 50 →
  x > y →
  A * y = B * x →
  x = 3 ∧ y = 2 := by sorry

end lcm_sum_ratio_problem_l2473_247311


namespace score_ordering_l2473_247317

-- Define the set of people
inductive Person : Type
| M : Person  -- Marty
| Q : Person  -- Quay
| S : Person  -- Shana
| Z : Person  -- Zane
| K : Person  -- Kaleana

-- Define a function to represent the score of each person
variable (score : Person → ℕ)

-- Define the conditions from the problem
def marty_condition (score : Person → ℕ) : Prop :=
  ∃ p : Person, score Person.M > score p

def quay_condition (score : Person → ℕ) : Prop :=
  score Person.Q = score Person.K

def shana_condition (score : Person → ℕ) : Prop :=
  ∃ p : Person, score Person.S < score p

def zane_condition (score : Person → ℕ) : Prop :=
  (score Person.Z < score Person.S) ∨ (score Person.Z > score Person.M)

-- Theorem statement
theorem score_ordering (score : Person → ℕ) :
  marty_condition score →
  quay_condition score →
  shana_condition score →
  zane_condition score →
  (score Person.Z < score Person.S) ∧
  (score Person.S < score Person.Q) ∧
  (score Person.Q < score Person.M) :=
sorry

end score_ordering_l2473_247317


namespace sphere_pyramid_height_l2473_247380

/-- The height of a square pyramid of spheres -/
def pyramid_height (n : ℕ) : ℝ :=
  2 * (n - 1)

/-- Theorem: The height of a square pyramid of spheres with radius 1,
    where the base layer has n^2 spheres and each subsequent layer has
    (n-1)^2 spheres until the top layer with 1 sphere, is 2(n-1). -/
theorem sphere_pyramid_height (n : ℕ) (h : n > 0) :
  let base_layer := n^2
  let top_layer := 1
  let sphere_radius := 1
  pyramid_height n = 2 * (n - 1) := by
  sorry

end sphere_pyramid_height_l2473_247380


namespace tree_spacing_l2473_247373

/-- Given a yard of length 400 meters with 26 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 16 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) :
  yard_length = 400 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 16 := by
sorry

end tree_spacing_l2473_247373


namespace triangle_side_length_l2473_247344

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (ha : t.a = 4) 
  (hb : t.b = 5) 
  (hS : t.S = 5 * Real.sqrt 3) :
  t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end triangle_side_length_l2473_247344


namespace largest_common_term_correct_l2473_247393

/-- First arithmetic progression with initial term 4 and common difference 5 -/
def ap1 (n : ℕ) : ℕ := 4 + 5 * n

/-- Second arithmetic progression with initial term 5 and common difference 10 -/
def ap2 (n : ℕ) : ℕ := 5 + 10 * n

/-- Predicate to check if a number is in both arithmetic progressions -/
def isCommonTerm (x : ℕ) : Prop :=
  ∃ n m : ℕ, ap1 n = x ∧ ap2 m = x

/-- The largest common term less than 300 -/
def largestCommonTerm : ℕ := 299

theorem largest_common_term_correct :
  isCommonTerm largestCommonTerm ∧
  largestCommonTerm < 300 ∧
  ∀ x : ℕ, isCommonTerm x → x < 300 → x ≤ largestCommonTerm :=
by sorry

#check largest_common_term_correct

end largest_common_term_correct_l2473_247393


namespace right_triangle_leg_square_l2473_247308

theorem right_triangle_leg_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = a + 2 →        -- Given condition
  b^2 = 4*(a + 1) := by
sorry

end right_triangle_leg_square_l2473_247308


namespace f_properties_l2473_247351

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| - |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a ≤ 0) :
  -- Part 1: Solution set when a = 0
  (a = 0 → {x : ℝ | f 0 x < 1} = {x : ℝ | 0 < x ∧ x < 2}) ∧
  -- Part 2: Range of a when triangle area > 3/2
  (∃ (x y : ℝ), x < y ∧ 
    (1/2 * (y - x) * (max (f a x) (f a y))) > 3/2 → a < -1) :=
by sorry

end f_properties_l2473_247351


namespace k_negative_sufficient_not_necessary_l2473_247396

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop := k * (1 - k) < 0

-- State the theorem
theorem k_negative_sufficient_not_necessary :
  (∀ k : ℝ, k < 0 → is_hyperbola k) ∧
  (∃ k : ℝ, is_hyperbola k ∧ ¬(k < 0)) :=
sorry

end k_negative_sufficient_not_necessary_l2473_247396


namespace map_scale_theorem_l2473_247375

/-- Represents the scale of a map as a ratio of 1 to some natural number. -/
structure MapScale where
  ratio : ℕ
  property : ratio > 0

/-- Calculates the map scale given the real distance and the corresponding map distance. -/
def calculate_map_scale (real_distance : ℕ) (map_distance : ℕ) : MapScale :=
  { ratio := real_distance / map_distance
    property := sorry }

theorem map_scale_theorem (real_km : ℕ) (map_cm : ℕ) 
  (h1 : real_km = 30) (h2 : map_cm = 20) : 
  (calculate_map_scale (real_km * 100000) map_cm).ratio = 150000 := by
  sorry

#check map_scale_theorem

end map_scale_theorem_l2473_247375


namespace subtraction_addition_result_l2473_247348

/-- The result of subtracting 567.89 from 1234.56 and then adding 300.30 is equal to 966.97 -/
theorem subtraction_addition_result : 
  (1234.56 - 567.89 + 300.30 : ℚ) = 966.97 := by sorry

end subtraction_addition_result_l2473_247348


namespace second_knife_set_price_l2473_247371

/-- Calculates the price of the second set of knives based on given sales data --/
def price_of_second_knife_set (
  houses_per_day : ℕ)
  (buy_percentage : ℚ)
  (first_set_price : ℕ)
  (weekly_sales : ℕ)
  (work_days : ℕ) : ℚ :=
  let buyers_per_day : ℚ := houses_per_day * buy_percentage
  let first_set_buyers_per_day : ℚ := buyers_per_day / 2
  let first_set_sales_per_day : ℚ := first_set_buyers_per_day * first_set_price
  let first_set_sales_per_week : ℚ := first_set_sales_per_day * work_days
  let second_set_sales_per_week : ℚ := weekly_sales - first_set_sales_per_week
  let second_set_buyers_per_week : ℚ := first_set_buyers_per_day * work_days
  second_set_sales_per_week / second_set_buyers_per_week

/-- Theorem stating that the price of the second set of knives is $150 --/
theorem second_knife_set_price :
  price_of_second_knife_set 50 (1/5) 50 5000 5 = 150 := by
  sorry

end second_knife_set_price_l2473_247371


namespace min_value_tangent_line_circle_l2473_247353

theorem min_value_tangent_line_circle (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 ≥ 2) → 
  (∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, x + y + a' = 0 → (x - b')^2 + (y - 1)^2 ≥ 2) → 
    (∃ x y : ℝ, x + y + a' = 0 ∧ (x - b')^2 + (y - 1)^2 = 2) → 
    (3 - 2*b)^2 / (2*a) ≤ (3 - 2*b')^2 / (2*a')) → 
  (3 - 2*b)^2 / (2*a) = 4 := by
sorry

end min_value_tangent_line_circle_l2473_247353


namespace debby_spent_14_tickets_l2473_247356

/-- The number of tickets Debby spent on a hat -/
def hat_tickets : ℕ := 2

/-- The number of tickets Debby spent on a stuffed animal -/
def stuffed_animal_tickets : ℕ := 10

/-- The number of tickets Debby spent on a yoyo -/
def yoyo_tickets : ℕ := 2

/-- The total number of tickets Debby spent -/
def total_tickets : ℕ := hat_tickets + stuffed_animal_tickets + yoyo_tickets

theorem debby_spent_14_tickets : total_tickets = 14 := by
  sorry

end debby_spent_14_tickets_l2473_247356


namespace triangle_side_length_l2473_247326

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = Real.sqrt 3 →
  c = 3 →
  B = 30 * π / 180 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos B →
  a = Real.sqrt 3 := by
sorry

end triangle_side_length_l2473_247326


namespace sum_x_y_equals_one_l2473_247389

theorem sum_x_y_equals_one (x y : ℝ) 
  (eq1 : x + 2*y = 1) 
  (eq2 : 2*x + y = 2) : 
  x + y = 1 := by
sorry

end sum_x_y_equals_one_l2473_247389


namespace square_difference_square_sum_mental_math_strategy_l2473_247303

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) :=
  sorry

theorem square_sum (n : ℕ) : (n + 1)^2 = n^2 + (2*n + 1) :=
  sorry

theorem mental_math_strategy :
  49^2 = 50^2 - 99 ∧ 51^2 = 50^2 + 101 :=
by
  have h1 : 49^2 = 50^2 - 99 := by
    calc
      49^2 = (50 - 1)^2 := by rfl
      _ = 50^2 - (2*50 - 1) := by apply square_difference
      _ = 50^2 - 99 := by ring
  
  have h2 : 51^2 = 50^2 + 101 := by
    calc
      51^2 = (50 + 1)^2 := by rfl
      _ = 50^2 + (2*50 + 1) := by apply square_sum
      _ = 50^2 + 101 := by ring
  
  exact ⟨h1, h2⟩

#check mental_math_strategy

end square_difference_square_sum_mental_math_strategy_l2473_247303


namespace coin_flip_probability_l2473_247370

theorem coin_flip_probability : ∃ p : ℝ, 
  p > 0 ∧ p < 1 ∧ 
  p^2 + (1-p)^2 = 4*p*(1-p) ∧
  ∀ q : ℝ, (q > 0 ∧ q < 1 ∧ q^2 + (1-q)^2 = 4*q*(1-q)) → q ≤ p ∧
  p = (3 + Real.sqrt 3) / 6 := by
sorry

end coin_flip_probability_l2473_247370


namespace perfect_square_powers_of_two_l2473_247345

theorem perfect_square_powers_of_two :
  (∃! (n : ℕ), ∃ (k : ℕ), 2^n + 3 = k^2) ∧
  (∃! (n : ℕ), ∃ (k : ℕ), 2^n + 1 = k^2) :=
by
  constructor
  · -- Proof for 2^n + 3
    sorry
  · -- Proof for 2^n + 1
    sorry

#check perfect_square_powers_of_two

end perfect_square_powers_of_two_l2473_247345


namespace unique_intersection_l2473_247339

/-- The function f(x) = 4 - 2x + x^2 -/
def f (x : ℝ) : ℝ := 4 - 2*x + x^2

/-- The function g(x) = 2 + 2x + x^2 -/
def g (x : ℝ) : ℝ := 2 + 2*x + x^2

theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    f p.1 = g p.1 ∧ 
    p = (1/2, 13/4) := by
  sorry

#check unique_intersection

end unique_intersection_l2473_247339


namespace cube_difference_as_sum_of_squares_l2473_247391

theorem cube_difference_as_sum_of_squares (n : ℤ) :
  (n + 2)^3 - n^3 = n^2 + (n + 2)^2 + (2*n + 2)^2 := by
  sorry

end cube_difference_as_sum_of_squares_l2473_247391


namespace max_x_squared_y_l2473_247363

theorem max_x_squared_y (x y : ℕ+) (h : 7 * x.val + 4 * y.val = 140) :
  ∀ (a b : ℕ+), 7 * a.val + 4 * b.val = 140 → x.val^2 * y.val ≥ a.val^2 * b.val :=
by sorry

end max_x_squared_y_l2473_247363


namespace subtracted_value_l2473_247362

theorem subtracted_value (n v : ℝ) (h1 : n = -10) (h2 : 2 * n - v = -12) : v = -8 := by
  sorry

end subtracted_value_l2473_247362


namespace max_remainder_l2473_247341

theorem max_remainder (m : ℕ) (n : ℕ) : 
  0 < m → m < 2015 → 2015 % m = n → n ≤ 1007 := by
  sorry

end max_remainder_l2473_247341


namespace gcd_36_54_l2473_247392

theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l2473_247392


namespace relationship_between_exponents_l2473_247349

theorem relationship_between_exponents 
  (m p t q : ℝ) 
  (n r s u : ℕ) 
  (h1 : (m^n)^2 = p^r)
  (h2 : p^r = t)
  (h3 : p^s = (m^u)^3)
  (h4 : (m^u)^3 = q)
  : 3 * u * r = 2 * n * s := by
  sorry

end relationship_between_exponents_l2473_247349


namespace bob_cleaning_time_l2473_247342

theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) : 
  alice_time = 40 → bob_fraction = 3 / 4 → bob_fraction * alice_time = 30 := by
  sorry

end bob_cleaning_time_l2473_247342


namespace intersection_of_A_and_B_l2473_247323

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x + 1) > 0}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≥ 1} := by sorry

end intersection_of_A_and_B_l2473_247323


namespace children_getting_on_bus_l2473_247324

theorem children_getting_on_bus (initial : ℝ) (got_off : ℝ) (final : ℝ) 
  (h1 : initial = 42.5)
  (h2 : got_off = 21.3)
  (h3 : final = 35.8) :
  final - (initial - got_off) = 14.6 := by
  sorry

end children_getting_on_bus_l2473_247324


namespace line_generates_surface_l2473_247321

-- Define the parabolas and the plane
def parabola1 (x y z : ℝ) : Prop := y^2 = 2*x ∧ z = 0
def parabola2 (x y z : ℝ) : Prop := 3*x = z^2 ∧ y = 0
def plane (y z : ℝ) : Prop := y = z

-- Define a line parallel to the plane y = z
def parallel_line (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ × ℝ), p ∈ L → q ∈ L → plane p.2.1 p.2.2 = plane q.2.1 q.2.2

-- Define the intersection of the line with the parabolas
def intersects_parabolas (L : Set (ℝ × ℝ × ℝ)) : Prop :=
  (∃ p ∈ L, parabola1 p.1 p.2.1 p.2.2) ∧ (∃ q ∈ L, parabola2 q.1 q.2.1 q.2.2)

-- The main theorem
theorem line_generates_surface (L : Set (ℝ × ℝ × ℝ)) :
  parallel_line L → intersects_parabolas L →
  ∀ (x y z : ℝ), (x, y, z) ∈ L → x = (y - z) * (y/2 - z/3) :=
sorry

end line_generates_surface_l2473_247321


namespace remainder_98_102_div_8_l2473_247316

theorem remainder_98_102_div_8 : (98 * 102) % 8 = 4 := by
  sorry

end remainder_98_102_div_8_l2473_247316


namespace lcm_factor_theorem_l2473_247381

theorem lcm_factor_theorem (A B : ℕ) (hcf lcm X : ℕ) : 
  A > 0 → B > 0 → 
  A = 368 → 
  hcf = Nat.gcd A B → 
  hcf = 23 → 
  lcm = Nat.lcm A B → 
  lcm = hcf * X * 16 → 
  X = 1 := by
sorry

end lcm_factor_theorem_l2473_247381


namespace sum_of_odd_numbers_l2473_247352

theorem sum_of_odd_numbers (N : ℕ) : 
  1001 + 1003 + 1005 + 1007 + 1009 = 5050 - N → N = 25 := by
  sorry

end sum_of_odd_numbers_l2473_247352


namespace fruit_bowl_problem_l2473_247343

/-- Proves that given a bowl with 14 apples and an unknown number of oranges,
    if removing 15 oranges results in apples being 70% of the remaining fruit,
    then the initial number of oranges was 21. -/
theorem fruit_bowl_problem (initial_oranges : ℕ) : 
  (14 : ℝ) / ((14 : ℝ) + (initial_oranges - 15)) = 0.7 → initial_oranges = 21 :=
by sorry

end fruit_bowl_problem_l2473_247343


namespace inequality_system_solution_l2473_247369

theorem inequality_system_solution (x : ℝ) : 
  3 * x > x + 6 ∧ (1/2) * x < -x + 5 → 3 < x ∧ x < 10/3 := by
  sorry

end inequality_system_solution_l2473_247369


namespace relationship_abc_l2473_247377

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sin (145 * π / 180) →
  b = Real.cos (52 * π / 180) →
  c = Real.tan (47 * π / 180) →
  a < b ∧ b < c := by sorry

end relationship_abc_l2473_247377


namespace percentage_of_m_l2473_247376

theorem percentage_of_m (j k l m : ℝ) : 
  (1.25 * j = 0.25 * k) →
  (1.5 * k = 0.5 * l) →
  (∃ p, 1.75 * l = p / 100 * m) →
  (0.2 * m = 7 * j) →
  (∃ p, 1.75 * l = p / 100 * m ∧ p = 75) := by
sorry

end percentage_of_m_l2473_247376


namespace dot_product_range_l2473_247313

/-- A circle centered at the origin and tangent to the line x-√3y=4 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The line x-√3y=4 -/
def TangentLine := {p : ℝ × ℝ | p.1 - Real.sqrt 3 * p.2 = 4}

/-- Point A where the circle intersects the negative x-axis -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B where the circle intersects the positive x-axis -/
def B : ℝ × ℝ := (2, 0)

/-- A point P inside the circle satisfying the geometric sequence condition -/
def P := {p : ℝ × ℝ | p ∈ Circle ∧ p.1^2 = p.2^2 + 2}

/-- The dot product of PA and PB -/
def dotProduct (p : ℝ × ℝ) : ℝ := 
  (A.1 - p.1) * (B.1 - p.1) + (A.2 - p.2) * (B.2 - p.2)

theorem dot_product_range :
  ∀ p ∈ P, -2 ≤ dotProduct p ∧ dotProduct p < 0 := by sorry

end dot_product_range_l2473_247313


namespace females_without_daughters_l2473_247386

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_females : ℕ
  daughters_with_children : ℕ

/-- The actual family structure of Bertha -/
def berthas_family : BerthaFamily :=
  { daughters := 8
  , granddaughters := 40
  , total_females := 48
  , daughters_with_children := 5 }

/-- Theorem stating the number of females without daughters in Bertha's family -/
theorem females_without_daughters (b : BerthaFamily) (h1 : b = berthas_family) :
  b.daughters + b.granddaughters - b.daughters_with_children = 43 := by
  sorry

#check females_without_daughters

end females_without_daughters_l2473_247386


namespace monthly_fixed_costs_correct_l2473_247384

/-- Represents the monthly fixed costs for producing electronic components -/
def monthly_fixed_costs : ℝ := 16399.50

/-- Represents the cost to produce one electronic component -/
def production_cost : ℝ := 80

/-- Represents the shipping cost for one electronic component -/
def shipping_cost : ℝ := 4

/-- Represents the number of components produced and sold monthly -/
def monthly_sales : ℕ := 150

/-- Represents the lowest selling price per component without loss -/
def break_even_price : ℝ := 193.33

/-- Theorem stating that the monthly fixed costs are correct given the other parameters -/
theorem monthly_fixed_costs_correct :
  monthly_fixed_costs = 
    monthly_sales * break_even_price - 
    monthly_sales * (production_cost + shipping_cost) :=
by sorry

end monthly_fixed_costs_correct_l2473_247384


namespace stock_change_theorem_l2473_247310

theorem stock_change_theorem (x : ℝ) (h : x > 0) :
  let day1_value := x * (1 - 0.3)
  let day2_value := day1_value * (1 + 0.5)
  (day2_value - x) / x = 0.05 := by sorry

end stock_change_theorem_l2473_247310


namespace yellow_ball_probability_l2473_247366

-- Define the number of red and yellow balls
def num_red_balls : ℕ := 3
def num_yellow_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_yellow_balls

-- Define the probability of selecting a yellow ball
def prob_yellow : ℚ := num_yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 4 / 7 := by
  sorry

end yellow_ball_probability_l2473_247366


namespace inserted_numbers_sum_l2473_247300

theorem inserted_numbers_sum (x y : ℝ) : 
  10 < x ∧ x < y ∧ y < 39 ∧  -- x and y are between 10 and 39
  (x / 10 = y / x) ∧         -- 10, x, y form a geometric sequence
  (y - x = 39 - y) →         -- x, y, 39 form an arithmetic sequence
  x + y = 11.25 :=           -- sum of x and y is 11¼
by sorry

end inserted_numbers_sum_l2473_247300


namespace similar_triangles_height_l2473_247357

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 15 :=
by sorry

end similar_triangles_height_l2473_247357


namespace water_added_calculation_l2473_247315

def initial_volume : ℝ := 340
def water_percentage : ℝ := 0.88
def cola_percentage : ℝ := 0.05
def sugar_percentage : ℝ := 1 - water_percentage - cola_percentage
def added_sugar : ℝ := 3.2
def added_cola : ℝ := 6.8
def final_sugar_percentage : ℝ := 0.075

theorem water_added_calculation (water_added : ℝ) : 
  (sugar_percentage * initial_volume + added_sugar) / 
  (initial_volume + added_sugar + added_cola + water_added) = final_sugar_percentage → 
  water_added = 10 := by
  sorry

end water_added_calculation_l2473_247315


namespace three_dollar_neg_one_l2473_247387

def dollar_op (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem three_dollar_neg_one : dollar_op 3 (-1) = 0 := by
  sorry

end three_dollar_neg_one_l2473_247387


namespace complement_of_union_l2473_247354

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def M : Set ℕ := {1,3,5,7}
def N : Set ℕ := {5,6,7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end complement_of_union_l2473_247354


namespace honey_purchase_cost_l2473_247328

def honey_problem (bulk_price min_spend tax_rate excess_pounds : ℕ) : Prop :=
  let min_pounds : ℕ := min_spend / bulk_price
  let total_pounds : ℕ := min_pounds + excess_pounds
  let pre_tax_cost : ℕ := total_pounds * bulk_price
  let tax_amount : ℕ := total_pounds * tax_rate
  let total_cost : ℕ := pre_tax_cost + tax_amount
  total_cost = 240

theorem honey_purchase_cost :
  honey_problem 5 40 1 32 := by sorry

end honey_purchase_cost_l2473_247328


namespace shape_cutting_theorem_l2473_247365

/-- Represents a cell in the shape --/
inductive Cell
| Black
| Gray

/-- Represents the shape as a list of cells --/
def Shape := List Cell

/-- A function to count the number of ways to cut the shape --/
def count_cuts (shape : Shape) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem shape_cutting_theorem (shape : Shape) :
  shape.length = 17 →
  count_cuts shape = 10 :=
sorry

end shape_cutting_theorem_l2473_247365


namespace count_primes_between_50_and_70_l2473_247395

theorem count_primes_between_50_and_70 : 
  (Finset.filter Nat.Prime (Finset.range 19)).card = 4 := by
  sorry

end count_primes_between_50_and_70_l2473_247395


namespace trigonometric_equality_l2473_247329

theorem trigonometric_equality : 
  1 / Real.cos (40 * π / 180) - 2 * Real.sqrt 3 / Real.sin (40 * π / 180) = -4 * Real.tan (20 * π / 180) := by
  sorry

end trigonometric_equality_l2473_247329


namespace binary_product_theorem_l2473_247374

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- The main theorem stating that the product of the given binary numbers equals the expected result -/
theorem binary_product_theorem :
  let a := [false, false, true, true, false, true]  -- 101100₂
  let b := [true, true, true]                       -- 111₂
  let c := [false, true]                            -- 10₂
  let result := [false, false, true, false, true, true, false, false, true]  -- 100110100₂
  binary_to_decimal a * binary_to_decimal b * binary_to_decimal c = binary_to_decimal result := by
  sorry


end binary_product_theorem_l2473_247374


namespace smallest_n_containing_all_binary_l2473_247383

/-- Given a natural number n, returns true if the binary representation of 1/n
    contains the binary representations of all numbers from 1 to 1990 as
    contiguous substrings after the decimal point. -/
def containsAllBinaryRepresentations (n : ℕ) : Prop := sorry

/-- Theorem stating that 2053 is the smallest natural number satisfying
    the condition of containing all binary representations from 1 to 1990. -/
theorem smallest_n_containing_all_binary : ∀ n : ℕ,
  n < 2053 → ¬(containsAllBinaryRepresentations n) ∧ containsAllBinaryRepresentations 2053 :=
sorry

end smallest_n_containing_all_binary_l2473_247383


namespace carnival_booth_rent_calculation_l2473_247398

def carnival_booth_rent (daily_popcorn_revenue : ℕ)
                        (cotton_candy_multiplier : ℕ)
                        (activity_days : ℕ)
                        (ingredient_cost : ℕ)
                        (total_earnings_after_expenses : ℕ) : Prop :=
  let daily_cotton_candy_revenue := daily_popcorn_revenue * cotton_candy_multiplier
  let total_revenue := (daily_popcorn_revenue + daily_cotton_candy_revenue) * activity_days
  let rent := total_revenue - ingredient_cost - total_earnings_after_expenses
  rent = 30

theorem carnival_booth_rent_calculation :
  carnival_booth_rent 50 3 5 75 895 := by
  sorry

end carnival_booth_rent_calculation_l2473_247398


namespace student_marks_average_l2473_247320

theorem student_marks_average (P C M B : ℝ) 
  (h1 : P + C + M + B = P + B + 180)
  (h2 : P = 1.20 * B) :
  (C + M) / 2 = 90 := by
  sorry

end student_marks_average_l2473_247320


namespace munchausen_polygon_exists_l2473_247332

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define a line as a set of points satisfying a linear equation
def Line : Type := Set (ℝ × ℝ)

-- Define what it means for a point to be inside a polygon
def inside (p : Point) (poly : Polygon) : Prop := sorry

-- Define what it means for a line to divide a polygon
def divides (l : Line) (poly : Polygon) : Prop := sorry

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Count the number of polygons resulting from dividing a polygon by a line
def count_divisions (l : Line) (poly : Polygon) : ℕ := sorry

-- The main theorem
theorem munchausen_polygon_exists :
  ∃ (P : Polygon) (O : Point),
    inside O P ∧
    ∀ (L : Line), passes_through L O →
      count_divisions L P = 3 := by sorry

end munchausen_polygon_exists_l2473_247332


namespace remainder_theorem_l2473_247364

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end remainder_theorem_l2473_247364


namespace sqrt_product_equality_l2473_247319

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2473_247319


namespace det_A_l2473_247337

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, -2; 5, 6, -4; 3, 3, 7]

theorem det_A : Matrix.det A = 168 := by sorry

end det_A_l2473_247337


namespace smallest_value_l2473_247355

theorem smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2 :=
by sorry

end smallest_value_l2473_247355


namespace max_vacation_savings_l2473_247399

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remaining_money (savings : ℕ) (ticket_cost : ℕ) : ℕ :=
  base8_to_base10 savings - ticket_cost

theorem max_vacation_savings :
  remaining_money 5273 1500 = 1247 := by sorry

end max_vacation_savings_l2473_247399


namespace cylinder_cut_face_area_l2473_247390

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a cut through the cylinder -/
structure CylinderCut (c : Cylinder) where
  arcAngle : ℝ  -- Angle between the two points on the circular face

/-- The area of the rectangular face resulting from the cut -/
def cutFaceArea (c : Cylinder) (cut : CylinderCut c) : ℝ :=
  c.height * (2 * c.radius)

theorem cylinder_cut_face_area 
  (c : Cylinder) 
  (cut : CylinderCut c) 
  (h_radius : c.radius = 4) 
  (h_height : c.height = 10) 
  (h_angle : cut.arcAngle = π) : 
  cutFaceArea c cut = 80 := by
  sorry

#eval (80 : ℤ) + (0 : ℤ) + (1 : ℤ)  -- Should evaluate to 81

end cylinder_cut_face_area_l2473_247390


namespace buddy_fraction_l2473_247347

theorem buddy_fraction (t s : ℚ) 
  (h1 : t > 0) 
  (h2 : s > 0) 
  (h3 : (1/4) * t = (3/5) * s) : 
  ((1/4) * t + (3/5) * s) / (t + s) = 6/17 := by
sorry

end buddy_fraction_l2473_247347


namespace max_value_on_curves_l2473_247314

theorem max_value_on_curves (m n x y : ℝ) (α β : ℝ) : 
  m = Real.sqrt 6 * Real.cos α →
  n = Real.sqrt 6 * Real.sin α →
  x = Real.sqrt 24 * Real.cos β →
  y = Real.sqrt 24 * Real.sin β →
  (∀ m' n' x' y' α' β', 
    m' = Real.sqrt 6 * Real.cos α' →
    n' = Real.sqrt 6 * Real.sin α' →
    x' = Real.sqrt 24 * Real.cos β' →
    y' = Real.sqrt 24 * Real.sin β' →
    m' * x' + n' * y' ≤ 12) ∧
  (∃ m' n' x' y' α' β', 
    m' = Real.sqrt 6 * Real.cos α' ∧
    n' = Real.sqrt 6 * Real.sin α' ∧
    x' = Real.sqrt 24 * Real.cos β' ∧
    y' = Real.sqrt 24 * Real.sin β' ∧
    m' * x' + n' * y' = 12) :=
by sorry

end max_value_on_curves_l2473_247314


namespace sphere_in_cube_ratios_l2473_247312

/-- The ratio of volumes and surface areas for a sphere inscribed in a cube -/
theorem sphere_in_cube_ratios (s : ℝ) (h : s > 0) :
  let sphere_volume := (4 / 3) * Real.pi * s^3
  let cube_volume := (2 * s)^3
  let sphere_surface_area := 4 * Real.pi * s^2
  let cube_surface_area := 6 * (2 * s)^2
  (sphere_volume / cube_volume = Real.pi / 6) ∧
  (sphere_surface_area / cube_surface_area = Real.pi / 6) := by
  sorry

end sphere_in_cube_ratios_l2473_247312


namespace congruence_solution_l2473_247325

theorem congruence_solution (n : ℤ) : 13 * n ≡ 9 [ZMOD 53] → n ≡ 17 [ZMOD 53] := by
  sorry

end congruence_solution_l2473_247325


namespace inequality_theorem_l2473_247379

theorem inequality_theorem (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_eq : a * b / (c * d) = (a + b) / (c + d)) :
  (a + b) * (c + d) ≥ (a + c) * (b + d) := by
sorry

end inequality_theorem_l2473_247379


namespace expression_simplification_l2473_247327

theorem expression_simplification :
  Real.sqrt 3 + Real.sqrt (3 + 5) + Real.sqrt (3 + 5 + 7) + Real.sqrt (3 + 5 + 7 + 9) =
  Real.sqrt 3 + 2 * Real.sqrt 2 + Real.sqrt 15 + 2 * Real.sqrt 6 := by
  sorry

end expression_simplification_l2473_247327


namespace geometric_sequence_fifth_term_l2473_247388

/-- Given a geometric sequence {aₙ} where a₁ = 1/3 and 2a₂ = a₄, prove that a₅ = 4/3 -/
theorem geometric_sequence_fifth_term (a : ℕ → ℚ) (h1 : a 1 = 1/3) (h2 : 2 * a 2 = a 4) :
  a 5 = 4/3 := by
  sorry

end geometric_sequence_fifth_term_l2473_247388


namespace black_to_white_area_ratio_l2473_247318

/-- The ratio of black to white area in concentric circles with radii 2, 4, 6, and 8 -/
theorem black_to_white_area_ratio : Real := by
  -- Define the radii of the circles
  let r1 : Real := 2
  let r2 : Real := 4
  let r3 : Real := 6
  let r4 : Real := 8

  -- Define the areas of the circles
  let A1 : Real := Real.pi * r1^2
  let A2 : Real := Real.pi * r2^2
  let A3 : Real := Real.pi * r3^2
  let A4 : Real := Real.pi * r4^2

  -- Define the areas of the black and white regions
  let black_area : Real := A1 + (A3 - A2)
  let white_area : Real := (A2 - A1) + (A4 - A3)

  -- Prove that the ratio of black area to white area is 3/5
  have h : black_area / white_area = 3 / 5 := by sorry

  exact 3 / 5

end black_to_white_area_ratio_l2473_247318


namespace root_sum_equals_three_l2473_247330

-- Define the equations
def equation1 (x : ℝ) : Prop := x + Real.log x = 3
def equation2 (x : ℝ) : Prop := x + (10 : ℝ)^x = 3

-- State the theorem
theorem root_sum_equals_three :
  ∀ x₁ x₂ : ℝ, equation1 x₁ → equation2 x₂ → x₁ + x₂ = 3 := by
  sorry

end root_sum_equals_three_l2473_247330


namespace existence_of_primes_with_gcd_one_l2473_247372

theorem existence_of_primes_with_gcd_one (n : ℕ) (h1 : n > 6) (h2 : Even n) :
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ Nat.gcd (n - p) (n - q) = 1 := by
  sorry

end existence_of_primes_with_gcd_one_l2473_247372


namespace poly_descending_order_l2473_247333

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := 2 * x^2 * y - 3 * x^3 - x * y^3 + 1

/-- The polynomial arranged in descending order of x -/
def descending_poly (x y : ℝ) : ℝ := -3 * x^3 + 2 * x^2 * y - x * y^3 + 1

/-- Theorem stating that the original polynomial is equal to the descending order polynomial -/
theorem poly_descending_order : ∀ x y : ℝ, original_poly x y = descending_poly x y := by
  sorry

end poly_descending_order_l2473_247333


namespace parallelepiped_properties_l2473_247309

/-- Represents a parallelepiped with given properties -/
structure Parallelepiped where
  height : ℝ
  lateral_edge_projection : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculate the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : Parallelepiped) : ℝ := sorry

/-- Calculate the volume of the parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

/-- Theorem stating the properties of the specific parallelepiped -/
theorem parallelepiped_properties (p : Parallelepiped) 
  (h1 : p.height = 12)
  (h2 : p.lateral_edge_projection = 5)
  (h3 : p.rhombus_area = 24)
  (h4 : p.rhombus_diagonal = 8) : 
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end parallelepiped_properties_l2473_247309


namespace cloth_sale_price_l2473_247302

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Proves that the total selling price of 85 meters of cloth with a profit of $15 per meter and a cost price of $90 per meter is $8925. -/
theorem cloth_sale_price : totalSellingPrice 85 15 90 = 8925 := by
  sorry

end cloth_sale_price_l2473_247302


namespace complex_abs_value_l2473_247346

theorem complex_abs_value : Complex.abs (-3 + (8/5) * Complex.I) = 17/5 := by
  sorry

end complex_abs_value_l2473_247346


namespace complex_multiplication_result_l2473_247335

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication_result : i^2 * (1 + i) = -1 - i := by sorry

end complex_multiplication_result_l2473_247335


namespace diamond_brace_to_ring_ratio_l2473_247322

def total_worth : ℕ := 14000
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000

def diamond_brace_cost : ℕ := total_worth - (ring_cost + car_cost)

theorem diamond_brace_to_ring_ratio :
  diamond_brace_cost / ring_cost = 2 := by sorry

end diamond_brace_to_ring_ratio_l2473_247322


namespace sqrt_product_equality_l2473_247360

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15 := by
  sorry

end sqrt_product_equality_l2473_247360


namespace largest_group_size_l2473_247385

def round_fraction (n : ℕ) (d : ℕ) (x : ℕ) : ℕ :=
  (2 * n * x + d) / (2 * d)

theorem largest_group_size :
  ∀ x : ℕ, x ≤ 37 ↔
    round_fraction 1 2 x + round_fraction 1 3 x + round_fraction 1 5 x ≤ x + 1 ∧
    (∀ y : ℕ, y > x →
      round_fraction 1 2 y + round_fraction 1 3 y + round_fraction 1 5 y > y + 1) :=
by sorry

end largest_group_size_l2473_247385


namespace quadratic_distinct_roots_m_range_l2473_247336

theorem quadratic_distinct_roots_m_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   m * x^2 + (2*m + 1) * x + m = 0 ∧ 
   m * y^2 + (2*m + 1) * y + m = 0) ↔ 
  (m > -1/4 ∧ m ≠ 0) :=
sorry

end quadratic_distinct_roots_m_range_l2473_247336


namespace symmetry_probability_l2473_247358

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- The size of the square grid -/
def gridSize : Nat := 13

/-- The center point P -/
def centerPoint : GridPoint := ⟨7, 7⟩

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of points excluding the center point -/
def pointsExcludingCenter : Nat := totalPoints - 1

/-- Checks if a point is on a line of symmetry through the center -/
def isOnSymmetryLine (q : GridPoint) : Prop :=
  q.x = centerPoint.x ∨ 
  q.y = centerPoint.y ∨ 
  q.x - centerPoint.x = q.y - centerPoint.y ∨
  q.x - centerPoint.x = centerPoint.y - q.y

/-- The number of points on lines of symmetry (excluding the center) -/
def symmetricPoints : Nat := 48

/-- The theorem stating the probability of Q being on a line of symmetry -/
theorem symmetry_probability : 
  (symmetricPoints : ℚ) / pointsExcludingCenter = 2 / 7 := by sorry

end symmetry_probability_l2473_247358


namespace hyperbola_eccentricity_l2473_247334

/-- Given a hyperbola with asymptotes y = ±(3/4)x, its eccentricity is either 5/4 or 5/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  let e := c / a
  c^2 = a^2 + b^2 →
  e = 5 / 4 ∨ e = 5 / 3 := by
  sorry

end hyperbola_eccentricity_l2473_247334


namespace greatest_x_value_l2473_247307

theorem greatest_x_value (x : ℝ) : 
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) → x ≤ -3 :=
by sorry

end greatest_x_value_l2473_247307


namespace min_value_theorem_l2473_247304

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1 / (3 * a + 5 * b) + 1 / (3 * b + 5 * c) + 1 / (3 * c + 5 * a) ≥ 9 / 8 := by
  sorry

end min_value_theorem_l2473_247304


namespace year2018_is_WuXu_l2473_247301

/-- Represents the Ten Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Twelve Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary Cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- The Sexagenary Cycle -/
def SexagenaryCycle : List SexagenaryYear := sorry

/-- Function to get the next year in the Sexagenary Cycle -/
def nextYear (year : SexagenaryYear) : SexagenaryYear := sorry

/-- 2016 is the Bing Shen year -/
def year2016 : SexagenaryYear :=
  { stem := HeavenlyStem.Bing, branch := EarthlyBranch.Shen }

/-- Theorem: 2018 is the Wu Xu year in the Sexagenary Cycle -/
theorem year2018_is_WuXu :
  (nextYear (nextYear year2016)) = { stem := HeavenlyStem.Wu, branch := EarthlyBranch.Xu } := by
  sorry


end year2018_is_WuXu_l2473_247301


namespace items_left_in_cart_l2473_247306

def initial_items : ℕ := 18
def deleted_items : ℕ := 10

theorem items_left_in_cart : initial_items - deleted_items = 8 := by
  sorry

end items_left_in_cart_l2473_247306


namespace incorrect_conversion_l2473_247350

def base4_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 4 + (n % 10)

def base2_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 2 + (n % 10)

theorem incorrect_conversion :
  base4_to_decimal 31 ≠ base2_to_decimal 62 :=
sorry

end incorrect_conversion_l2473_247350
