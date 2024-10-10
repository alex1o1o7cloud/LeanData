import Mathlib

namespace committees_with_restriction_l4007_400798

def total_students : ℕ := 9
def committee_size : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committees_with_restriction (total : ℕ) (size : ℕ) : 
  total = total_students → size = committee_size → 
  (choose total size) - (choose (total - 2) (size - 2)) = 91 := by
  sorry

end committees_with_restriction_l4007_400798


namespace inequality_proof_l4007_400700

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2) := by
sorry

end inequality_proof_l4007_400700


namespace min_distance_ellipse_line_is_zero_l4007_400789

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and the line x - √2 y - 4 = 0 is 0. -/
theorem min_distance_ellipse_line_is_zero :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | p.1 - Real.sqrt 2 * p.2 - 4 = 0}
  (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ ellipse ∧ q ∈ line ∧ ‖p - q‖ = 0) :=
by sorry

end min_distance_ellipse_line_is_zero_l4007_400789


namespace perpendicular_condition_l4007_400786

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "in plane" relation for a line
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_condition 
  (l m n : Line) (α : Plane) 
  (h1 : in_plane m α) 
  (h2 : in_plane n α) :
  (perp_line_plane l α → perp_line_line l m ∧ perp_line_line l n) ∧ 
  ¬(perp_line_line l m ∧ perp_line_line l n → perp_line_plane l α) :=
sorry

end perpendicular_condition_l4007_400786


namespace restaurant_sales_problem_l4007_400733

/-- Represents the dinner sales for a restaurant over four days. -/
structure RestaurantSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions and proof goal for the restaurant sales problem. -/
theorem restaurant_sales_problem (sales : RestaurantSales) : 
  sales.monday = 40 →
  sales.tuesday = sales.monday + 40 →
  sales.wednesday = sales.tuesday / 2 →
  sales.thursday > sales.wednesday →
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 203 →
  sales.thursday - sales.wednesday = 3 := by
  sorry


end restaurant_sales_problem_l4007_400733


namespace min_value_expression_l4007_400736

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + 2*b^2 + 2/(a + 2*b)^2 ≥ 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + 2*b₀^2 + 2/(a₀ + 2*b₀)^2 = 2 :=
by sorry

end min_value_expression_l4007_400736


namespace isosceles_triangle_area_l4007_400725

/-- An isosceles triangle with height 8 and perimeter 32 has an area of 48 -/
theorem isosceles_triangle_area (b : ℝ) (l : ℝ) : 
  b > 0 → l > 0 →
  (2 * l + b = 32) →  -- perimeter condition
  (l^2 = (b/2)^2 + 8^2) →  -- Pythagorean theorem for height
  (1/2 * b * 8 = 48) :=  -- area formula
by sorry

end isosceles_triangle_area_l4007_400725


namespace probability_third_draw_defective_10_3_l4007_400754

/-- Given a set of products with some defective ones, this function calculates
    the probability of drawing a defective product on the third draw, given
    that the first draw was defective. -/
def probability_third_draw_defective (total_products : ℕ) (defective_products : ℕ) : ℚ :=
  if total_products < 3 ∨ defective_products < 1 ∨ defective_products > total_products then 0
  else
    let remaining_after_first := total_products - 1
    let defective_after_first := defective_products - 1
    let numerator := (remaining_after_first - defective_after_first) * defective_after_first +
                     defective_after_first * (defective_after_first - 1)
    let denominator := remaining_after_first * (remaining_after_first - 1)
    ↑numerator / ↑denominator

/-- Theorem stating that for 10 products with 3 defective ones, the probability
    of drawing a defective product on the third draw, given that the first
    draw was defective, is 2/9. -/
theorem probability_third_draw_defective_10_3 :
  probability_third_draw_defective 10 3 = 2 / 9 := by
  sorry

end probability_third_draw_defective_10_3_l4007_400754


namespace existence_of_special_integer_l4007_400771

theorem existence_of_special_integer : ∃ n : ℕ+, 
  (Nat.card {p : ℕ | Nat.Prime p ∧ p ∣ n} = 2000) ∧ 
  (n ∣ 2^(n : ℕ) + 1) := by
  sorry

end existence_of_special_integer_l4007_400771


namespace remaining_time_for_finger_exerciser_l4007_400788

theorem remaining_time_for_finger_exerciser 
  (total_time : Nat) 
  (piano_time : Nat) 
  (writing_time : Nat) 
  (reading_time : Nat) 
  (h1 : total_time = 120)
  (h2 : piano_time = 30)
  (h3 : writing_time = 25)
  (h4 : reading_time = 38) :
  total_time - (piano_time + writing_time + reading_time) = 27 := by
  sorry

end remaining_time_for_finger_exerciser_l4007_400788


namespace distribute_five_objects_l4007_400765

/-- The number of ways to distribute n distinguishable objects into 2 indistinguishable containers -/
def distribute_objects (n : ℕ) : ℕ :=
  (2^n - 2) / 2 + 2

/-- Theorem: There are 17 ways to distribute 5 distinguishable objects into 2 indistinguishable containers -/
theorem distribute_five_objects : distribute_objects 5 = 17 := by
  sorry

end distribute_five_objects_l4007_400765


namespace problem_1_l4007_400728

theorem problem_1 (a : ℚ) (h : a = 1/2) : 2*a^2 - 5*a + a^2 + 4*a - 3*a^2 - 2 = -5/2 := by
  sorry

end problem_1_l4007_400728


namespace maria_gave_65_towels_l4007_400705

/-- The number of towels Maria gave to her mother -/
def towels_given_to_mother (green_towels white_towels remaining_towels : ℕ) : ℕ :=
  green_towels + white_towels - remaining_towels

/-- Proof that Maria gave 65 towels to her mother -/
theorem maria_gave_65_towels :
  towels_given_to_mother 40 44 19 = 65 := by
  sorry

end maria_gave_65_towels_l4007_400705


namespace parallel_perpendicular_implication_l4007_400790

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

theorem parallel_perpendicular_implication 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β := sorry

end parallel_perpendicular_implication_l4007_400790


namespace initial_sum_calculation_l4007_400745

/-- The initial sum that earns a specific total simple interest over 4 years with varying interest rates -/
def initial_sum (total_interest : ℚ) (rate1 rate2 rate3 rate4 : ℚ) : ℚ :=
  total_interest / (rate1 + rate2 + rate3 + rate4)

/-- Theorem stating that given the specified conditions, the initial sum is 5000/9 -/
theorem initial_sum_calculation :
  initial_sum 100 (3/100) (5/100) (4/100) (6/100) = 5000/9 := by
  sorry

#eval initial_sum 100 (3/100) (5/100) (4/100) (6/100)

end initial_sum_calculation_l4007_400745


namespace perpendicular_line_through_point_l4007_400791

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The problem statement -/
theorem perpendicular_line_through_point :
  let givenLine : Line2D := { a := 2, b := 1, c := -3 }
  let pointA : Point2D := { x := 0, y := 4 }
  let resultLine : Line2D := { a := 1, b := -2, c := 8 }
  perpendicularLines givenLine resultLine ∧
  pointOnLine pointA resultLine := by
  sorry

end perpendicular_line_through_point_l4007_400791


namespace spade_calculation_l4007_400739

-- Define the ⬥ operation
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

-- State the theorem
theorem spade_calculation : spade 3 (spade 6 5) = -112 := by
  sorry

end spade_calculation_l4007_400739


namespace reasoning_method_is_inductive_l4007_400732

-- Define the set of animals
inductive Animal : Type
| Ape : Animal
| Cat : Animal
| Elephant : Animal
| OtherMammal : Animal

-- Define the breathing method
inductive BreathingMethod : Type
| Lungs : BreathingMethod

-- Define the reasoning method
inductive ReasoningMethod : Type
| Inductive : ReasoningMethod
| Deductive : ReasoningMethod
| Analogical : ReasoningMethod
| CompleteInductive : ReasoningMethod

-- Define a function that represents breathing for specific animals
def breathes : Animal → BreathingMethod
| Animal.Ape => BreathingMethod.Lungs
| Animal.Cat => BreathingMethod.Lungs
| Animal.Elephant => BreathingMethod.Lungs
| Animal.OtherMammal => BreathingMethod.Lungs

-- Define a predicate for reasoning from specific to general
def reasonsFromSpecificToGeneral (method : ReasoningMethod) : Prop :=
  method = ReasoningMethod.Inductive

-- Theorem statement
theorem reasoning_method_is_inductive :
  (∀ a : Animal, breathes a = BreathingMethod.Lungs) →
  (reasonsFromSpecificToGeneral ReasoningMethod.Inductive) :=
by sorry

end reasoning_method_is_inductive_l4007_400732


namespace mark_change_factor_l4007_400763

theorem mark_change_factor (n : ℕ) (original_avg new_avg : ℝ) (h1 : n = 12) (h2 : original_avg = 36) (h3 : new_avg = 72) :
  ∃ (factor : ℝ), factor * (n * original_avg) = n * new_avg ∧ factor = 2 := by
  sorry

end mark_change_factor_l4007_400763


namespace equal_digit_probability_l4007_400741

def num_dice : ℕ := 6
def sides_per_die : ℕ := 20
def one_digit_outcomes : ℕ := 9
def two_digit_outcomes : ℕ := 11

def prob_one_digit : ℚ := one_digit_outcomes / sides_per_die
def prob_two_digit : ℚ := two_digit_outcomes / sides_per_die

def equal_digit_prob : ℚ := (num_dice.choose (num_dice / 2)) *
  (prob_one_digit ^ (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2))

theorem equal_digit_probability :
  equal_digit_prob = 4851495 / 16000000 := by sorry

end equal_digit_probability_l4007_400741


namespace reeses_height_l4007_400721

theorem reeses_height (parker daisy reese : ℝ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : (parker + daisy + reese) / 3 = 64) :
  reese = 60 := by sorry

end reeses_height_l4007_400721


namespace vector_properties_l4007_400750

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

def M : ℝ × ℝ := (C.1 + 3*c.1, C.2 + 3*c.2)
def N : ℝ × ℝ := (C.1 - 2*b.1, C.2 - 2*b.2)

theorem vector_properties :
  (3*a.1 + b.1 - 3*c.1 = 6 ∧ 3*a.2 + b.2 - 3*c.2 = -42) ∧
  (a = (-b.1 - c.1, -b.2 - c.2)) ∧
  (M = (0, 20) ∧ N = (9, 2) ∧ (M.1 - N.1 = 9 ∧ M.2 - N.2 = -18)) :=
by sorry

end vector_properties_l4007_400750


namespace dan_licks_l4007_400735

/-- The number of licks it takes for each person to get to the center of a lollipop -/
structure LollipopLicks where
  michael : ℕ
  sam : ℕ
  david : ℕ
  lance : ℕ
  dan : ℕ

/-- The average number of licks for all five people -/
def average (l : LollipopLicks) : ℚ :=
  (l.michael + l.sam + l.david + l.lance + l.dan) / 5

/-- Theorem stating that Dan takes 58 licks to get to the center of a lollipop -/
theorem dan_licks (l : LollipopLicks) : 
  l.michael = 63 → l.sam = 70 → l.david = 70 → l.lance = 39 → average l = 60 → l.dan = 58 := by
  sorry

end dan_licks_l4007_400735


namespace y_intercept_for_specific_line_l4007_400746

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope 3 and x-intercept (4, 0), the y-intercept is (0, -12). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := 3, x_intercept := (4, 0) }
  y_intercept l = (0, -12) := by sorry

end y_intercept_for_specific_line_l4007_400746


namespace expression_simplification_l4007_400766

theorem expression_simplification (q : ℚ) : 
  ((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6) = 76 * q - 44 := by
  sorry

end expression_simplification_l4007_400766


namespace geometric_series_first_term_l4007_400783

theorem geometric_series_first_term (a₁ q : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → ∃ (aₙ : ℝ), aₙ = a₁ * q^(n-1)) →  -- Geometric series condition
  (-1 < q) →                                         -- Convergence condition
  (q < 1) →                                          -- Convergence condition
  (q ≠ 0) →                                          -- Non-zero common ratio
  (a₁ / (1 - q) = 1) →                               -- Sum of series is 1
  (|a₁| / (1 - |q|) = 2) →                           -- Sum of absolute values is 2
  a₁ = 4/3 := by
sorry

end geometric_series_first_term_l4007_400783


namespace triangle_area_from_rectangle_l4007_400706

/-- The area of one right triangle formed by cutting a rectangle diagonally --/
theorem triangle_area_from_rectangle (length width : Real) (h_length : length = 0.5) (h_width : width = 0.3) :
  (length * width) / 2 = 0.075 := by
  sorry

end triangle_area_from_rectangle_l4007_400706


namespace complex_equation_solution_l4007_400773

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * Real.sqrt 3 + 3 * Complex.I) * z = Complex.I * Real.sqrt 3 →
  z = (Real.sqrt 3 / 4 : ℂ) + (Complex.I / 4) :=
by sorry

end complex_equation_solution_l4007_400773


namespace sequence_correctness_l4007_400797

def a (n : ℕ) : ℤ := (-1 : ℤ)^(n + 1) * n^2

theorem sequence_correctness : 
  (a 1 = 1) ∧ (a 2 = -4) ∧ (a 3 = 9) ∧ (a 4 = -16) ∧ (a 5 = 25) := by
  sorry

end sequence_correctness_l4007_400797


namespace increasing_quadratic_condition_l4007_400784

/-- If f(x) = x^2 + 2(a - 1)x + 2 is an increasing function on the interval (4, +∞), then a ≥ -3 -/
theorem increasing_quadratic_condition (a : ℝ) : 
  (∀ x > 4, Monotone (fun x => x^2 + 2*(a - 1)*x + 2)) → a ≥ -3 := by
sorry

end increasing_quadratic_condition_l4007_400784


namespace min_value_shifted_l4007_400748

/-- A quadratic function f(x) with a minimum value of 2 -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

/-- The function g(x) which is f(x-2015) -/
def g (c : ℝ) (x : ℝ) : ℝ := f c (x - 2015)

theorem min_value_shifted (c : ℝ) (h : ∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ ∃ (x₀ : ℝ), f c x₀ = m) 
  (hmin : ∃ (x₀ : ℝ), f c x₀ = 2) :
  ∃ (m : ℝ), ∀ (x : ℝ), g c x ≥ m ∧ ∃ (x₀ : ℝ), g c x₀ = m ∧ m = 2 :=
sorry

end min_value_shifted_l4007_400748


namespace max_value_cos_sin_l4007_400755

theorem max_value_cos_sin (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end max_value_cos_sin_l4007_400755


namespace min_visible_sum_l4007_400778

/-- Represents a die in the cube -/
structure Die where
  faces : Fin 6 → Nat
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the large 4x4x4 cube -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum sum of visible faces -/
theorem min_visible_sum (cube : LargeCube) : 
  visibleSum cube ≥ 144 :=
sorry

end min_visible_sum_l4007_400778


namespace modified_lucas_60th_term_mod_5_l4007_400758

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modified_lucas_60th_term_mod_5 :
  modifiedLucas 59 % 5 = 1 := by
  sorry

end modified_lucas_60th_term_mod_5_l4007_400758


namespace quadratic_inequality_range_l4007_400729

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_inequality_range_l4007_400729


namespace ravi_mobile_price_l4007_400782

/-- The purchase price of Ravi's mobile phone -/
def mobile_price : ℝ :=
  -- Define the variable for the mobile phone price
  sorry

/-- The selling price of the refrigerator -/
def fridge_sell_price : ℝ :=
  15000 * (1 - 0.04)

/-- The selling price of the mobile phone -/
def mobile_sell_price : ℝ :=
  mobile_price * 1.10

/-- The total selling price of both items -/
def total_sell_price : ℝ :=
  fridge_sell_price + mobile_sell_price

/-- The total purchase price of both items plus profit -/
def total_purchase_plus_profit : ℝ :=
  15000 + mobile_price + 200

theorem ravi_mobile_price :
  (total_sell_price = total_purchase_plus_profit) →
  mobile_price = 6000 :=
by sorry

end ravi_mobile_price_l4007_400782


namespace solution_proof_l4007_400727

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 13*x - 6

/-- The largest real solution to the equation -/
noncomputable def n : ℝ := 13 + Real.sqrt 61

/-- The decomposition of n into d + √(e + √f) -/
def d : ℕ := 13
def e : ℕ := 61
def f : ℕ := 0

theorem solution_proof :
  equation n ∧ 
  n = d + Real.sqrt (e + Real.sqrt f) ∧
  d + e + f = 74 := by sorry

end solution_proof_l4007_400727


namespace bubble_gum_cost_l4007_400722

/-- Given a number of bubble gum pieces and a total cost in cents,
    calculate the cost per piece of bubble gum. -/
def cost_per_piece (num_pieces : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / num_pieces

/-- Theorem stating that the cost per piece of bubble gum is 18 cents
    given the specific conditions of the problem. -/
theorem bubble_gum_cost :
  cost_per_piece 136 2448 = 18 := by
  sorry

end bubble_gum_cost_l4007_400722


namespace stripe_area_theorem_l4007_400749

/-- Represents a cylindrical silo -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Represents a stripe wrapped around a cylinder -/
structure Stripe where
  width : ℝ
  revolutions : ℕ

/-- Calculates the area of a stripe wrapped around a cylinder -/
def stripeArea (c : Cylinder) (s : Stripe) : ℝ :=
  s.width * c.height

theorem stripe_area_theorem (c : Cylinder) (s : Stripe) :
  stripeArea c s = s.width * c.height := by sorry

end stripe_area_theorem_l4007_400749


namespace solution_set_nonempty_iff_m_in_range_l4007_400787

open Set

theorem solution_set_nonempty_iff_m_in_range (m : ℝ) :
  (∃ x : ℝ, |x - m| + |x + 2| < 4) ↔ m ∈ Ioo (-6) 2 := by
  sorry

end solution_set_nonempty_iff_m_in_range_l4007_400787


namespace remainder_3973_div_28_l4007_400769

theorem remainder_3973_div_28 : 3973 % 28 = 9 := by sorry

end remainder_3973_div_28_l4007_400769


namespace jills_total_earnings_l4007_400770

/-- Calculates Jill's earnings over three months given her work schedule --/
def jills_earnings (first_month_daily_rate : ℕ) (days_per_month : ℕ) : ℕ :=
  let first_month := first_month_daily_rate * days_per_month
  let second_month := (2 * first_month_daily_rate) * days_per_month
  let third_month := (2 * first_month_daily_rate) * (days_per_month / 2)
  first_month + second_month + third_month

/-- Theorem stating that Jill's earnings over three months equal $1200 --/
theorem jills_total_earnings :
  jills_earnings 10 30 = 1200 := by
  sorry

end jills_total_earnings_l4007_400770


namespace lcm_23_46_827_l4007_400703

theorem lcm_23_46_827 (h1 : 46 = 23 * 2) (h2 : Nat.Prime 827) :
  Nat.lcm 23 (Nat.lcm 46 827) = 38042 :=
by sorry

end lcm_23_46_827_l4007_400703


namespace irrational_arithmetic_properties_l4007_400756

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (x : ℝ) = q)

-- Theorem statement
theorem irrational_arithmetic_properties :
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ IsIrrational (a + b)) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ ∃ (q : ℚ), (a - b : ℝ) = q) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ ∃ (q : ℚ), (a * b : ℝ) = q) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ b ≠ 0 ∧ ∃ (q : ℚ), (a / b : ℝ) = q) :=
by sorry

end irrational_arithmetic_properties_l4007_400756


namespace largest_whole_number_less_than_150_over_9_l4007_400796

theorem largest_whole_number_less_than_150_over_9 :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x < 150 :=
by sorry

end largest_whole_number_less_than_150_over_9_l4007_400796


namespace triangle_third_side_l4007_400775

theorem triangle_third_side (a b h : ℝ) (ha : a = 25) (hb : b = 30) (hh : h = 24) :
  ∃ c, (c = 25 ∨ c = 11) ∧ 
  (∃ s, s * h = a * b ∧ 
   ((c + s) * (c - s) = a^2 - b^2 ∨ (c + s) * (c - s) = b^2 - a^2)) :=
by sorry

end triangle_third_side_l4007_400775


namespace trigonometric_identity_l4007_400704

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a^2 + b^2) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = (a^7 + b^7) / (a^2 + b^2)^6 :=
by sorry

end trigonometric_identity_l4007_400704


namespace range_of_a_l4007_400743

-- Define the condition that |x-3|+|x+5|>a holds for any x ∈ ℝ
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 3| + |x + 5| > a

-- State the theorem
theorem range_of_a :
  {a : ℝ | condition a} = Set.Iio 8 :=
by sorry

end range_of_a_l4007_400743


namespace flowerbed_fence_length_l4007_400781

/-- Calculates the perimeter of a rectangular flowerbed with given width and length rule -/
def flowerbed_perimeter (width : ℝ) : ℝ :=
  let length := 2 * width - 1
  2 * (width + length)

/-- Theorem stating that a rectangular flowerbed with width 4 meters and length 1 meter less than twice its width has a perimeter of 22 meters -/
theorem flowerbed_fence_length : flowerbed_perimeter 4 = 22 := by
  sorry

end flowerbed_fence_length_l4007_400781


namespace m_increasing_range_l4007_400708

def f (x : ℝ) : ℝ := (x - 1)^2

def is_m_increasing (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x ∈ D, x + m ∈ D ∧ f (x + m) ≥ f x

theorem m_increasing_range (m : ℝ) :
  is_m_increasing f m (Set.Ici 0) → m ∈ Set.Ici 2 := by
  sorry

end m_increasing_range_l4007_400708


namespace min_value_absolute_difference_l4007_400761

theorem min_value_absolute_difference (x : ℝ) :
  ((2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2) →
  (∃ y : ℝ, y = |x - 1| - |x + 3| ∧ 
   (∀ z : ℝ, ((2 * z - 1) / 3 - 1 ≥ z - (5 - 3 * z) / 2) → y ≤ |z - 1| - |z + 3|) ∧
   y = -2 - 8 / 11) :=
by sorry

end min_value_absolute_difference_l4007_400761


namespace johnny_guitar_picks_l4007_400710

theorem johnny_guitar_picks (total_picks : ℕ) (red_picks blue_picks yellow_picks : ℕ) : 
  total_picks = red_picks + blue_picks + yellow_picks →
  2 * red_picks = total_picks →
  3 * blue_picks = total_picks →
  blue_picks = 12 →
  yellow_picks = 6 := by
sorry

end johnny_guitar_picks_l4007_400710


namespace two_digit_number_puzzle_l4007_400768

theorem two_digit_number_puzzle :
  ∀ (x : ℕ),
  x < 10 →
  let original := 21 * x
  let reversed := 12 * x
  original < 100 →
  original - reversed = 27 →
  original = 63 := by
sorry

end two_digit_number_puzzle_l4007_400768


namespace min_c_value_l4007_400712

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2010 ∧
    p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|)) :
  1006 ≤ c.val := by sorry

end min_c_value_l4007_400712


namespace twin_prime_power_theorem_l4007_400752

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q = p + 2

theorem twin_prime_power_theorem :
  ∀ n : ℕ, (∃ p q : ℕ, is_twin_prime p q ∧ is_twin_prime (2^n + p) (2^n + q)) ↔ n = 1 ∨ n = 3 :=
sorry

end twin_prime_power_theorem_l4007_400752


namespace M_subset_N_l4007_400713

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l4007_400713


namespace largest_n_for_sin_cos_inequality_l4007_400715

theorem largest_n_for_sin_cos_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * ↑n)) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / (2 * ↑m)) ∧
  n = 8 :=
sorry

end largest_n_for_sin_cos_inequality_l4007_400715


namespace hundredth_term_equals_981_l4007_400759

/-- Sequence of powers of 3 or sums of distinct powers of 3 -/
def PowerOf3Sequence : ℕ → ℕ :=
  sorry

/-- The 100th term of the PowerOf3Sequence -/
def HundredthTerm : ℕ := PowerOf3Sequence 100

theorem hundredth_term_equals_981 : HundredthTerm = 981 := by
  sorry

end hundredth_term_equals_981_l4007_400759


namespace pet_store_gerbils_l4007_400772

/-- The initial number of gerbils in a pet store -/
def initial_gerbils : ℕ := 68

/-- The number of gerbils sold -/
def sold_gerbils : ℕ := 14

/-- The difference between the initial number and the number sold -/
def difference : ℕ := 54

theorem pet_store_gerbils : 
  initial_gerbils = sold_gerbils + difference := by sorry

end pet_store_gerbils_l4007_400772


namespace circumscribed_sphere_surface_area_l4007_400709

theorem circumscribed_sphere_surface_area (cube_volume : ℝ) (h : cube_volume = 64) :
  let cube_side := cube_volume ^ (1/3)
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius ^ 2 = 48 * Real.pi :=
by
  sorry

#check circumscribed_sphere_surface_area

end circumscribed_sphere_surface_area_l4007_400709


namespace complement_of_A_l4007_400753

/-- Given that the universal set U is the set of real numbers and 
    A is the set of real numbers x such that 1 < x ≤ 3,
    prove that the complement of A with respect to U 
    is the set of real numbers x such that x ≤ 1 or x > 3 -/
theorem complement_of_A (U : Set ℝ) (A : Set ℝ) 
  (h_U : U = Set.univ)
  (h_A : A = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  U \ A = {x : ℝ | x ≤ 1 ∨ x > 3} := by
  sorry

end complement_of_A_l4007_400753


namespace k_range_given_a_n_geq_a_3_l4007_400730

/-- A sequence where a_n = n^2 - k*n -/
def a (n : ℕ+) (k : ℝ) : ℝ := n.val^2 - k * n.val

/-- The theorem stating that if a_n ≥ a_3 for all positive integers n, then k is in [5, 7] -/
theorem k_range_given_a_n_geq_a_3 :
  (∀ n : ℕ+, a n k ≥ a 3 k) → k ∈ Set.Icc (5 : ℝ) 7 := by
  sorry

end k_range_given_a_n_geq_a_3_l4007_400730


namespace prob_one_of_three_wins_l4007_400777

/-- The probability that one of three mutually exclusive events occurs is the sum of their individual probabilities -/
theorem prob_one_of_three_wins (pX pY pZ : ℚ) 
  (hX : pX = 1/6) (hY : pY = 1/10) (hZ : pZ = 1/8) : 
  pX + pY + pZ = 47/120 := by
  sorry

end prob_one_of_three_wins_l4007_400777


namespace specific_arithmetic_sequence_sum_l4007_400742

/-- Sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a₁ aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem stating the sum of the specific arithmetic sequence -/
theorem specific_arithmetic_sequence_sum :
  arithmeticSequenceSum 1000 5000 4 = 3003000 := by
  sorry

#eval arithmeticSequenceSum 1000 5000 4

end specific_arithmetic_sequence_sum_l4007_400742


namespace inequality_proof_l4007_400720

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  (2 + x)^2 / (1 + x)^2 + (2 + y)^2 / (1 + y)^2 ≥ 9/2 := by
  sorry

end inequality_proof_l4007_400720


namespace integral_extrema_l4007_400767

open Real MeasureTheory

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ∫ t in (x - a)..(x + a), sqrt (4 * a^2 - t^2)

theorem integral_extrema (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, |x| ≤ a → f a x ≤ 2 * π * a^2) ∧
  (∀ x : ℝ, |x| ≤ a → f a x ≥ π * a^2) ∧
  (∃ x : ℝ, |x| ≤ a ∧ f a x = 2 * π * a^2) ∧
  (∃ x : ℝ, |x| ≤ a ∧ f a x = π * a^2) :=
sorry

end integral_extrema_l4007_400767


namespace inverse_proportion_problem_l4007_400794

/-- Given that x and y are inversely proportional, prove that when x = -12, y = -56.25 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 60) (h3 : x = 3 * y) :
  x = -12 → y = -56.25 := by
  sorry

end inverse_proportion_problem_l4007_400794


namespace jar_water_problem_l4007_400799

theorem jar_water_problem (capacity_x : ℝ) (capacity_y : ℝ) 
  (h1 : capacity_y = (1 / 2) * capacity_x) 
  (h2 : capacity_x > 0) :
  let initial_water_x := (1 / 2) * capacity_x
  let initial_water_y := (1 / 2) * capacity_y
  let final_water_x := initial_water_x + initial_water_y
  final_water_x = (3 / 4) * capacity_x := by
sorry

end jar_water_problem_l4007_400799


namespace base4_product_l4007_400744

-- Define a function to convert from base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

-- Define a function to convert from decimal to base 4
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the two base 4 numbers
def num1 : List Nat := [1, 3, 2]  -- 132₄
def num2 : List Nat := [1, 2]     -- 12₄

-- State the theorem
theorem base4_product :
  decimalToBase4 (base4ToDecimal num1 * base4ToDecimal num2) = [2, 3, 1, 0] := by
  sorry

end base4_product_l4007_400744


namespace first_digit_is_one_l4007_400795

def base_three_number : List Nat := [1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 2, 2, 1, 0, 1, 2, 2, 1, 0, 2]

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

def decimal_to_base_nine (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

theorem first_digit_is_one :
  (decimal_to_base_nine (base_three_to_decimal base_three_number)).head? = some 1 := by
  sorry

end first_digit_is_one_l4007_400795


namespace square_plus_one_geq_two_abs_l4007_400780

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_geq_two_abs_l4007_400780


namespace line_through_circle_center_l4007_400723

/-- The line equation -/
def line_eq (x y : ℝ) (a : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center (x y : ℝ) : Prop :=
  circle_eq x y ∧ ∀ x' y', circle_eq x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

/-- The main theorem -/
theorem line_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, circle_center x y ∧ line_eq x y a) → a = 1 :=
sorry

end line_through_circle_center_l4007_400723


namespace pure_imaginary_condition_l4007_400764

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ b : ℝ, (m^2 + Complex.I) * (1 + m * Complex.I) = Complex.I * b) → m = 0 ∨ m = 1 := by
sorry

end pure_imaginary_condition_l4007_400764


namespace f_properties_l4007_400714

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℤ := floor ((x + 1) / 3 - floor (x / 3))

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (x + 3) = f x) ∧ 
  (∀ y : ℤ, y ∈ Set.range f → y = 0 ∨ y = 1) ∧
  (∀ y : ℤ, y = 0 ∨ y = 1 → ∃ x : ℝ, f x = y) :=
by sorry

end f_properties_l4007_400714


namespace friends_signed_up_first_day_l4007_400716

/-- The number of friends who signed up on the first day -/
def friends_first_day : ℕ := sorry

/-- The total number of friends who signed up (including first day and rest of the week) -/
def total_friends : ℕ := friends_first_day + 7

/-- The total money earned by Katrina and her friends -/
def total_money : ℕ := 125

theorem friends_signed_up_first_day : 
  5 + 5 * total_friends + 5 * total_friends = total_money ∧ friends_first_day = 5 := by sorry

end friends_signed_up_first_day_l4007_400716


namespace scientific_notation_of_1500000_l4007_400726

theorem scientific_notation_of_1500000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500000 = a * (10 : ℝ) ^ n ∧ a = 1.5 ∧ n = 6 := by
  sorry

end scientific_notation_of_1500000_l4007_400726


namespace reggie_loses_by_21_points_l4007_400747

/-- Represents the types of basketball shots -/
inductive ShotType
  | Layup
  | FreeThrow
  | ThreePointer
  | HalfCourt

/-- Returns the point value for a given shot type -/
def pointValue (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.Layup => 1
  | ShotType.FreeThrow => 2
  | ShotType.ThreePointer => 3
  | ShotType.HalfCourt => 5

/-- Calculates the total points for a set of shots -/
def totalPoints (layups freeThrows threePointers halfCourt : ℕ) : ℕ :=
  layups * pointValue ShotType.Layup +
  freeThrows * pointValue ShotType.FreeThrow +
  threePointers * pointValue ShotType.ThreePointer +
  halfCourt * pointValue ShotType.HalfCourt

/-- Theorem stating the difference in points between Reggie's brother and Reggie -/
theorem reggie_loses_by_21_points :
  totalPoints 3 2 5 4 - totalPoints 4 3 2 1 = 21 := by
  sorry

#eval totalPoints 3 2 5 4 - totalPoints 4 3 2 1

end reggie_loses_by_21_points_l4007_400747


namespace equal_intercept_line_equation_l4007_400751

/-- A line passing through (1, 2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through (1, 2)
  point_condition : 2 = k * (1 - 1) + 2
  -- The line has equal intercepts on both axes
  equal_intercepts : 2 - k = 1 - 2 / k

/-- The equation of the line is either x + y - 3 = 0 or 2x - y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, x + y - 3 = 0 ↔ y - 2 = l.k * (x - 1)) ∨
  (∀ x y, 2 * x - y = 0 ↔ y - 2 = l.k * (x - 1)) :=
sorry

end equal_intercept_line_equation_l4007_400751


namespace log_6_15_in_terms_of_a_b_l4007_400757

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm with arbitrary base
noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statement
theorem log_6_15_in_terms_of_a_b (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log 6 15 = (b + 1 - a) / (a + b) := by
  sorry


end log_6_15_in_terms_of_a_b_l4007_400757


namespace target_heart_rate_for_sprinting_is_156_l4007_400760

-- Define the athlete's age
def age : ℕ := 30

-- Define the maximum heart rate calculation
def max_heart_rate (a : ℕ) : ℕ := 225 - a

-- Define the target heart rate for jogging
def target_heart_rate_jogging (mhr : ℕ) : ℕ := (mhr * 3) / 4

-- Define the target heart rate for sprinting
def target_heart_rate_sprinting (thr_jogging : ℕ) : ℕ := thr_jogging + 10

-- Theorem to prove
theorem target_heart_rate_for_sprinting_is_156 : 
  target_heart_rate_sprinting (target_heart_rate_jogging (max_heart_rate age)) = 156 := by
  sorry

end target_heart_rate_for_sprinting_is_156_l4007_400760


namespace tangent_line_and_extrema_l4007_400792

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  let tangent_line (x : ℝ) := 1
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f 0) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ f (Real.pi / 2)) ∧
  (HasDerivAt f (tangent_line 0 - f 0) 0) :=
sorry

end tangent_line_and_extrema_l4007_400792


namespace set_equality_implies_sum_l4007_400738

theorem set_equality_implies_sum (a b : ℝ) : 
  ({4, a} : Set ℝ) = ({2, a * b} : Set ℝ) → a + b = 4 := by
  sorry

end set_equality_implies_sum_l4007_400738


namespace alexander_payment_l4007_400774

/-- The cost of tickets at an amusement park -/
def ticket_cost (child_cost adult_cost : ℕ) (alexander_child alexander_adult anna_child anna_adult : ℕ) : Prop :=
  let alexander_total := child_cost * alexander_child + adult_cost * alexander_adult
  let anna_total := child_cost * anna_child + adult_cost * anna_adult
  (child_cost = 600) ∧
  (alexander_child = 2) ∧
  (alexander_adult = 3) ∧
  (anna_child = 3) ∧
  (anna_adult = 2) ∧
  (alexander_total = anna_total + 200)

theorem alexander_payment :
  ∀ (child_cost adult_cost : ℕ),
  ticket_cost child_cost adult_cost 2 3 3 2 →
  child_cost * 2 + adult_cost * 3 = 3600 :=
by
  sorry

end alexander_payment_l4007_400774


namespace line_tangent_to_parabola_l4007_400707

/-- The line 4x + 3y + 9 = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 3 * y + 9 = 0 ∧ y^2 = 16 * x := by
  sorry

end line_tangent_to_parabola_l4007_400707


namespace three_in_M_l4007_400762

def U : Set ℤ := {x | x^2 - 6*x < 0}

theorem three_in_M (M : Set ℤ) (h : (U \ M) = {1, 2}) : 3 ∈ M := by
  sorry

end three_in_M_l4007_400762


namespace rolls_bought_l4007_400724

theorem rolls_bought (price_per_dozen : ℝ) (money_spent : ℝ) (rolls_per_dozen : ℕ) : 
  price_per_dozen = 5 → money_spent = 15 → rolls_per_dozen = 12 → 
  (money_spent / price_per_dozen) * rolls_per_dozen = 36 :=
by
  sorry

end rolls_bought_l4007_400724


namespace rationalize_denominator_l4007_400740

theorem rationalize_denominator : 
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = 
  (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42 := by sorry

end rationalize_denominator_l4007_400740


namespace product_zero_l4007_400711

theorem product_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end product_zero_l4007_400711


namespace union_equals_reals_l4007_400701

def S : Set ℝ := {x | (x - 2)^2 > 9}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

theorem union_equals_reals (a : ℝ) : S ∪ T a = Set.univ ↔ a ∈ Set.Ioo (-3) (-1) := by
  sorry

end union_equals_reals_l4007_400701


namespace inequality_solution_l4007_400719

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x < -4 ∨ x ≥ 2) :=
sorry

end inequality_solution_l4007_400719


namespace candy_comparison_l4007_400793

/-- Represents a person with their candy bags -/
structure Person where
  name : String
  bags : List Nat

/-- Calculates the total candy for a person -/
def totalCandy (p : Person) : Nat :=
  p.bags.sum

theorem candy_comparison (sandra roger emily : Person)
  (h_sandra : sandra.bags = [6, 6])
  (h_roger : roger.bags = [11, 3])
  (h_emily : emily.bags = [4, 7, 5]) :
  totalCandy emily > totalCandy roger ∧
  totalCandy roger > totalCandy sandra ∧
  totalCandy sandra = 12 := by
  sorry

#eval totalCandy { name := "Sandra", bags := [6, 6] }
#eval totalCandy { name := "Roger", bags := [11, 3] }
#eval totalCandy { name := "Emily", bags := [4, 7, 5] }

end candy_comparison_l4007_400793


namespace ophelias_current_age_l4007_400776

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- Given Lennon's current age and the relationship between Ophelia and Lennon's ages in two years,
    prove that Ophelia's current age is 38 years. -/
theorem ophelias_current_age 
  (lennon ophelia : Person)
  (lennon_current_age : lennon.age = 8)
  (future_age_relation : ophelia.age + 2 = 4 * (lennon.age + 2)) :
  ophelia.age = 38 := by
  sorry

end ophelias_current_age_l4007_400776


namespace first_nonzero_digit_of_1_over_113_l4007_400737

theorem first_nonzero_digit_of_1_over_113 : ∃ (n : ℕ) (r : ℚ), 
  (1 : ℚ) / 113 = n / 10 + r ∧ 
  0 < r ∧ 
  r < 1 / 10 ∧ 
  n = 8 := by
sorry

end first_nonzero_digit_of_1_over_113_l4007_400737


namespace stratified_sample_size_l4007_400779

theorem stratified_sample_size 
  (total_population : ℕ) 
  (elderly_population : ℕ) 
  (elderly_sample : ℕ) 
  (n : ℕ) 
  (h1 : total_population = 162) 
  (h2 : elderly_population = 27) 
  (h3 : elderly_sample = 6) 
  (h4 : elderly_population * n = total_population * elderly_sample) : 
  n = 36 := by
sorry

end stratified_sample_size_l4007_400779


namespace inequality_range_l4007_400702

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end inequality_range_l4007_400702


namespace exists_valid_road_configuration_l4007_400785

/-- A configuration of roads connecting four villages at the vertices of a square -/
structure RoadConfiguration where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The total length of roads in the configuration -/
  total_length : ℝ
  /-- Ensure that all villages are connected -/
  all_connected : Bool

/-- Theorem stating that there exists a valid road configuration with total length less than 5.5 km -/
theorem exists_valid_road_configuration :
  ∃ (config : RoadConfiguration),
    config.side_length = 2 ∧
    config.all_connected = true ∧
    config.total_length < 5.5 := by
  sorry

end exists_valid_road_configuration_l4007_400785


namespace box_tape_relation_l4007_400731

def tape_needed (long_side short_side : ℝ) (num_boxes : ℕ) : ℝ :=
  num_boxes * (long_side + 2 * short_side)

theorem box_tape_relation (L S : ℝ) :
  tape_needed L S 5 + tape_needed 40 40 2 = 540 →
  L = 60 - 2 * S :=
by
  sorry

end box_tape_relation_l4007_400731


namespace circle_polar_equation_l4007_400718

/-- The polar equation ρ = 2cosθ represents a circle with center at (1,0) and radius 1 -/
theorem circle_polar_equation :
  ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ ↔
  (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1 :=
by sorry

end circle_polar_equation_l4007_400718


namespace stewart_farm_sheep_count_l4007_400734

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
    sheep / horses = 4 / 7 →
    horses * 230 = 12880 →
    sheep = 32 :=
by
  sorry

end stewart_farm_sheep_count_l4007_400734


namespace combined_annual_income_after_expenses_l4007_400717

def brady_income : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
def dwayne_income : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_expense : ℕ := 450
def dwayne_expense : ℕ := 300

theorem combined_annual_income_after_expenses :
  (brady_income.sum - brady_expense) + (dwayne_income.sum - dwayne_expense) = 3930 := by
  sorry

end combined_annual_income_after_expenses_l4007_400717
