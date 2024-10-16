import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_completion_l1777_177725

theorem quadratic_completion (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 36 = (x + m)^2 + 20) → 
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_l1777_177725


namespace NUMINAMATH_CALUDE_max_area_specific_quadrilateral_l1777_177776

/-- A convex quadrilateral with given side lengths -/
structure ConvexQuadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  convex : ab > 0 ∧ bc > 0 ∧ cd > 0 ∧ da > 0

/-- The area of a convex quadrilateral -/
def area (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem: Maximum area of a specific convex quadrilateral -/
theorem max_area_specific_quadrilateral :
  ∃ (q : ConvexQuadrilateral),
    q.ab = 2 ∧ q.bc = 4 ∧ q.cd = 5 ∧ q.da = 3 ∧
    ∀ (q' : ConvexQuadrilateral),
      q'.ab = 2 → q'.bc = 4 → q'.cd = 5 → q'.da = 3 →
      area q' ≤ 2 * Real.sqrt 30 :=
  sorry

end NUMINAMATH_CALUDE_max_area_specific_quadrilateral_l1777_177776


namespace NUMINAMATH_CALUDE_apple_price_l1777_177778

/-- The price of apples given Emmy's and Gerry's money and the total number of apples they can buy -/
theorem apple_price (emmy_money : ℝ) (gerry_money : ℝ) (total_apples : ℝ) 
  (h1 : emmy_money = 200)
  (h2 : gerry_money = 100)
  (h3 : total_apples = 150) :
  (emmy_money + gerry_money) / total_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_l1777_177778


namespace NUMINAMATH_CALUDE_sam_oatmeal_cookies_l1777_177755

/-- Given a total number of cookies and a ratio of three types of cookies,
    calculate the number of cookies of the second type. -/
def oatmealCookies (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  let totalParts := ratio1 + ratio2 + ratio3
  let cookiesPerPart := total / totalParts
  ratio2 * cookiesPerPart

/-- Theorem stating that given 36 total cookies and a ratio of 2:3:4,
    the number of oatmeal cookies is 12. -/
theorem sam_oatmeal_cookies :
  oatmealCookies 36 2 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sam_oatmeal_cookies_l1777_177755


namespace NUMINAMATH_CALUDE_chord_segment_ratio_l1777_177728

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a chord
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point

-- Define the intersection of two chords
def chord_intersection (c : Circle) (ch1 ch2 : Chord c) : Point := sorry

-- Power of a Point theorem
axiom power_of_point (c : Circle) (ch1 ch2 : Chord c) (q : Point) :
  let x := chord_intersection c ch1 ch2
  (x.1 - q.1) * (ch1.p2.1 - q.1) = (x.1 - q.1) * (ch2.p2.1 - q.1)

-- Main theorem
theorem chord_segment_ratio (c : Circle) (ch1 ch2 : Chord c) :
  let q := chord_intersection c ch1 ch2
  let x := ch1.p1
  let y := ch1.p2
  let w := ch2.p1
  let z := ch2.p2
  (x.1 - q.1) = 5 →
  (w.1 - q.1) = 7 →
  (y.1 - q.1) / (z.1 - q.1) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_segment_ratio_l1777_177728


namespace NUMINAMATH_CALUDE_si_o_bond_is_polar_covalent_l1777_177711

-- Define the electronegativity values
def electronegativity_Si : ℝ := 1.90
def electronegativity_O : ℝ := 3.44

-- Define the range for polar covalent bonds
def polar_covalent_lower_bound : ℝ := 0.5
def polar_covalent_upper_bound : ℝ := 1.7

-- Define a function to check if a bond is polar covalent
def is_polar_covalent (electronegativity_diff : ℝ) : Prop :=
  polar_covalent_lower_bound ≤ electronegativity_diff ∧
  electronegativity_diff ≤ polar_covalent_upper_bound

-- Theorem: The silicon-oxygen bonds in SiO2 are polar covalent
theorem si_o_bond_is_polar_covalent :
  is_polar_covalent (electronegativity_O - electronegativity_Si) :=
by
  sorry


end NUMINAMATH_CALUDE_si_o_bond_is_polar_covalent_l1777_177711


namespace NUMINAMATH_CALUDE_concentric_circles_area_l1777_177774

theorem concentric_circles_area (r₁ : ℝ) (chord_length : ℝ) (h₁ : r₁ = 50) (h₂ : chord_length = 120) : 
  let r₂ := Real.sqrt (r₁^2 + (chord_length/2)^2)
  π * (r₂^2 - r₁^2) = 3600 * π := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_l1777_177774


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1777_177726

theorem cubic_equation_roots : ∃ (z : ℂ), z^3 + z^2 - z = 7 + 7*I :=
by
  -- Prove that 4 + i and -3 - i are roots of the equation
  have h1 : (4 + I)^3 + (4 + I)^2 - (4 + I) = 7 + 7*I := by sorry
  have h2 : (-3 - I)^3 + (-3 - I)^2 - (-3 - I) = 7 + 7*I := by sorry
  
  -- Show that at least one of these roots satisfies the equation
  exact ⟨4 + I, h1⟩

-- Note: This theorem only proves the existence of one root,
-- but we know there are at least two roots satisfying the equation.

end NUMINAMATH_CALUDE_cubic_equation_roots_l1777_177726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1777_177759

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 5 + a 8 = 39) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1777_177759


namespace NUMINAMATH_CALUDE_triangle_type_l1777_177767

theorem triangle_type (A B C : ℝ) (BC AC : ℝ) (h : BC * Real.cos A = AC * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_type_l1777_177767


namespace NUMINAMATH_CALUDE_weight_replaced_is_75_l1777_177761

/-- The weight of the replaced person in a group, given the following conditions:
  * There are 7 persons initially
  * The average weight increases by 3.5 kg when a new person replaces one of them
  * The weight of the new person is 99.5 kg
-/
def weight_of_replaced_person (num_persons : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - (num_persons * avg_weight_increase)

/-- Theorem stating that the weight of the replaced person is 75 kg -/
theorem weight_replaced_is_75 :
  weight_of_replaced_person 7 3.5 99.5 = 75 := by sorry

end NUMINAMATH_CALUDE_weight_replaced_is_75_l1777_177761


namespace NUMINAMATH_CALUDE_original_integer_is_21_l1777_177768

theorem original_integer_is_21 (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a + b + c) / 3 + d = 29)
  (h2 : (a + b + d) / 3 + c = 23)
  (h3 : (a + c + d) / 3 + b = 21)
  (h4 : (b + c + d) / 3 + a = 17) :
  a = 21 ∨ b = 21 ∨ c = 21 ∨ d = 21 := by
  sorry

end NUMINAMATH_CALUDE_original_integer_is_21_l1777_177768


namespace NUMINAMATH_CALUDE_union_equals_interval_l1777_177764

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 5*x - 6 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Define the open interval (-6, 6)
def openInterval : Set ℝ := Ioo (-6) 6

-- Theorem statement
theorem union_equals_interval : A ∪ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_union_equals_interval_l1777_177764


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l1777_177791

/-- The decrease in area of an equilateral triangle when its sides are shortened --/
theorem equilateral_triangle_area_decrease :
  ∀ (s : ℝ),
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 100 * Real.sqrt 3 →
  let new_s := s - 3
  let original_area := (s^2 * Real.sqrt 3) / 4
  let new_area := (new_s^2 * Real.sqrt 3) / 4
  original_area - new_area = 27.75 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l1777_177791


namespace NUMINAMATH_CALUDE_beckys_fish_tank_l1777_177798

theorem beckys_fish_tank (current_amount desired_total : ℝ) 
  (h1 : current_amount = 7.75)
  (h2 : desired_total = 14.75) :
  desired_total - current_amount = 7 := by
  sorry

end NUMINAMATH_CALUDE_beckys_fish_tank_l1777_177798


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_19_l1777_177757

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_19 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 19 → n ≤ 9910 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_19_l1777_177757


namespace NUMINAMATH_CALUDE_german_team_goals_l1777_177715

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def exactlyTwoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  {x : ℕ | exactlyTwoCorrect x} = {11, 12, 14, 16, 17} := by sorry

end NUMINAMATH_CALUDE_german_team_goals_l1777_177715


namespace NUMINAMATH_CALUDE_range_of_m_l1777_177770

theorem range_of_m (m : ℝ) : 
  (∃! (n : ℕ), n = 4 ∧ (∀ x : ℤ, (m < x ∧ x < 4) ↔ (0 ≤ x ∧ x < 4))) → 
  (-1 ≤ m ∧ m < 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1777_177770


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1777_177784

theorem two_numbers_difference (x y : ℝ) (h_sum : x + y = 25) (h_product : x * y = 144) :
  |x - y| = 7 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1777_177784


namespace NUMINAMATH_CALUDE_polynomial_never_equals_33_l1777_177732

theorem polynomial_never_equals_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_never_equals_33_l1777_177732


namespace NUMINAMATH_CALUDE_calculate_small_orders_l1777_177749

/-- Given information about packing peanuts usage in orders, calculate the number of small orders. -/
theorem calculate_small_orders (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) (peanuts_per_small : ℕ) :
  total_peanuts = 800 →
  large_orders = 3 →
  peanuts_per_large = 200 →
  peanuts_per_small = 50 →
  (total_peanuts - large_orders * peanuts_per_large) / peanuts_per_small = 4 :=
by sorry

end NUMINAMATH_CALUDE_calculate_small_orders_l1777_177749


namespace NUMINAMATH_CALUDE_divisibility_by_1987_l1777_177772

def odd_product : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (2 * i + 1)) 1

def even_product : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (2 * i + 2)) 1

theorem divisibility_by_1987 : ∃ k : ℤ, (odd_product 993 + even_product 993 : ℤ) = k * 1987 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1987_l1777_177772


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1777_177738

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1777_177738


namespace NUMINAMATH_CALUDE_store_discount_l1777_177752

theorem store_discount (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) : 
  initial_discount = 0.40 →
  additional_discount = 0.10 →
  claimed_discount = 0.55 →
  let price_after_first_discount := 1 - initial_discount
  let price_after_second_discount := price_after_first_discount * (1 - additional_discount)
  let actual_discount := 1 - price_after_second_discount
  actual_discount = 0.46 ∧ claimed_discount - actual_discount = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_l1777_177752


namespace NUMINAMATH_CALUDE_kates_discount_is_eight_percent_l1777_177710

-- Define the bills and total paid
def bobs_bill : ℚ := 30
def kates_bill : ℚ := 25
def total_paid : ℚ := 53

-- Define the discount percentage
def discount_percentage : ℚ := (bobs_bill + kates_bill - total_paid) / kates_bill * 100

-- Theorem statement
theorem kates_discount_is_eight_percent :
  discount_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_kates_discount_is_eight_percent_l1777_177710


namespace NUMINAMATH_CALUDE_range_of_a_range_of_x_l1777_177744

-- Define the conditions p and q
def p (x : ℝ) : Prop := Real.sqrt (x - 1) ≤ 1
def q (x a : ℝ) : Prop := -1 ≤ x ∧ x ≤ a

-- Define the set A based on condition p
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the set B based on condition q
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

-- Theorem 1: Range of a when q is necessary but not sufficient for p
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬p x) ↔ 2 ≤ a :=
sorry

-- Theorem 2: Range of x when a = 1 and at least one of p or q holds true
theorem range_of_x : 
  ∀ x : ℝ, (p x ∨ q x 1) ↔ -1 ≤ x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_x_l1777_177744


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1777_177797

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4025.25 →
  rate = 9 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 8950 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1777_177797


namespace NUMINAMATH_CALUDE_collinear_vectors_magnitude_l1777_177730

/-- Given two planar vectors a and b that are collinear and have a negative dot product,
    prove that the magnitude of b is 2√2. -/
theorem collinear_vectors_magnitude (m : ℝ) :
  let a : ℝ × ℝ := (2 * m + 1, 3)
  let b : ℝ × ℝ := (2, m)
  (∃ (k : ℝ), a = k • b) →  -- collinearity condition
  (a.1 * b.1 + a.2 * b.2 < 0) →  -- dot product condition
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_magnitude_l1777_177730


namespace NUMINAMATH_CALUDE_simplify_fraction_l1777_177702

theorem simplify_fraction : 5 * (14 / 3) * (9 / -42) = -5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1777_177702


namespace NUMINAMATH_CALUDE_average_time_theorem_l1777_177785

def relay_race (y z w : ℝ) : Prop :=
  y = 58 ∧ z = 26 ∧ w = 2*z

theorem average_time_theorem (y z w : ℝ) (h : relay_race y z w) :
  (y + z + w) / 3 = (58 + 26 + 2*26) / 3 := by sorry

end NUMINAMATH_CALUDE_average_time_theorem_l1777_177785


namespace NUMINAMATH_CALUDE_cloth_loss_per_meter_l1777_177716

/-- Calculates the loss per meter of cloth given the total meters sold, total selling price, and cost price per meter. -/
def loss_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  let total_cost_price := total_meters * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_meters

/-- Theorem stating that for 400 meters of cloth sold at Rs. 18,000 with a cost price of Rs. 50 per meter, the loss per meter is Rs. 5. -/
theorem cloth_loss_per_meter :
  loss_per_meter 400 18000 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cloth_loss_per_meter_l1777_177716


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l1777_177795

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem first_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_prod : a 2 * a 3 * a 4 = 27) 
  (h_a7 : a 7 = 27) :
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l1777_177795


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1777_177700

/-- A line defined by the equation ax + (2-a)y + 1 = 0 -/
def line (a : ℝ) (x y : ℝ) : Prop := a * x + (2 - a) * y + 1 = 0

/-- The theorem states that for any real number a, 
    the line ax + (2-a)y + 1 = 0 passes through the point (-1/2, -1/2) -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line a (-1/2) (-1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1777_177700


namespace NUMINAMATH_CALUDE_area_of_EFGH_l1777_177739

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The configuration of rectangles forming EFGH -/
structure Configuration where
  small_rectangle : Rectangle
  large_rectangle : Rectangle

/-- The properties of the configuration as described in the problem -/
def valid_configuration (c : Configuration) : Prop :=
  c.small_rectangle.height = 6 ∧
  c.large_rectangle.width = 2 * c.small_rectangle.width ∧
  c.large_rectangle.height = 2 * c.small_rectangle.height

theorem area_of_EFGH (c : Configuration) (h : valid_configuration c) :
  c.large_rectangle.area = 144 :=
sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l1777_177739


namespace NUMINAMATH_CALUDE_total_earnings_l1777_177719

/-- The total earnings of Salvadore and Santo, given Salvadore's earnings and that Santo earned half of Salvadore's earnings -/
theorem total_earnings (salvadore_earnings : ℕ) (h : salvadore_earnings = 1956) :
  salvadore_earnings + (salvadore_earnings / 2) = 2934 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l1777_177719


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l1777_177707

def base_5_to_10 (b : ℕ) : ℕ := 780 * b

def base_c_to_10 (c : ℕ) : ℕ := 4 * (c + 1)

def valid_base_5_digit (b : ℕ) : Prop := 1 ≤ b ∧ b ≤ 4

def valid_base_c (c : ℕ) : Prop := c > 6

theorem smallest_sum_B_plus_c :
  ∃ (B c : ℕ),
    valid_base_5_digit B ∧
    valid_base_c c ∧
    base_5_to_10 B = base_c_to_10 c ∧
    (∀ (B' c' : ℕ),
      valid_base_5_digit B' →
      valid_base_c c' →
      base_5_to_10 B' = base_c_to_10 c' →
      B + c ≤ B' + c') ∧
    B + c = 195 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l1777_177707


namespace NUMINAMATH_CALUDE_work_completion_time_l1777_177724

theorem work_completion_time
  (A_work : ℝ)
  (B_work : ℝ)
  (C_work : ℝ)
  (h1 : A_work = 1 / 3)
  (h2 : B_work + C_work = 1 / 3)
  (h3 : B_work = 1 / 6)
  : 1 / (A_work + C_work) = 2 :=
by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l1777_177724


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l1777_177717

def diamond (x y : ℕ) : ℕ := Nat.gcd x y

def oplus (x y : ℕ) : ℕ := Nat.lcm x y

theorem gcd_lcm_problem : 
  (oplus (oplus (diamond 24 36) (diamond 54 24)) (diamond (48 * 60) (72 * 48))) = 576 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l1777_177717


namespace NUMINAMATH_CALUDE_point_b_position_l1777_177748

theorem point_b_position (a b : ℝ) : 
  a = -2 → (b - a = 4 ∨ a - b = 4) → (b = 2 ∨ b = -6) := by
  sorry

end NUMINAMATH_CALUDE_point_b_position_l1777_177748


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l1777_177789

variable (ξ : ℕ → ℝ)
variable (n : ℕ)
variable (p : ℝ)

-- ξ follows a binomial distribution B(n, p)
def is_binomial (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop := sorry

-- Expected value of ξ
def expectation (ξ : ℕ → ℝ) : ℝ := sorry

-- Variance of ξ
def variance (ξ : ℕ → ℝ) : ℝ := sorry

theorem binomial_distribution_parameters 
  (h1 : is_binomial ξ n p)
  (h2 : expectation ξ = 5/3)
  (h3 : variance ξ = 10/9) :
  n = 5 ∧ p = 1/3 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l1777_177789


namespace NUMINAMATH_CALUDE_uncrossed_numbers_count_l1777_177703

theorem uncrossed_numbers_count : 
  let total_numbers := 1000
  let gcd_value := Nat.gcd 1000 15
  let crossed_out := (total_numbers - 1) / gcd_value + 1
  total_numbers - crossed_out = 800 := by
  sorry

end NUMINAMATH_CALUDE_uncrossed_numbers_count_l1777_177703


namespace NUMINAMATH_CALUDE_fair_haired_employees_percentage_l1777_177750

theorem fair_haired_employees_percentage 
  (total_employees : ℕ) 
  (women_fair_hair_percentage : ℚ) 
  (fair_haired_women_percentage : ℚ) 
  (h1 : women_fair_hair_percentage = 10 / 100) 
  (h2 : fair_haired_women_percentage = 40 / 100) :
  (women_fair_hair_percentage * total_employees) / 
  (fair_haired_women_percentage * total_employees) = 25 / 100 := by
sorry

end NUMINAMATH_CALUDE_fair_haired_employees_percentage_l1777_177750


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l1777_177781

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 19500) → x = 53800 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l1777_177781


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l1777_177706

/-- Given a rectangle with length 12 and perimeter 36, prove that the ratio of its width to its length is 1:2 -/
theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 → -- width is positive
  12 > 0 → -- length is positive
  2 * w + 2 * 12 = 36 → -- perimeter formula
  w / 12 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l1777_177706


namespace NUMINAMATH_CALUDE_point_on_circle_l1777_177786

theorem point_on_circle (t : ℝ) :
  let x := (3 * t^2 - 1) / (t^2 + 3)
  let y := 6 * t / (t^2 + 3)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_circle_l1777_177786


namespace NUMINAMATH_CALUDE_puppy_price_calculation_l1777_177779

/-- Calculates the price per puppy in John's puppy selling scenario -/
theorem puppy_price_calculation (initial_puppies : ℕ) (stud_fee profit : ℚ) : 
  initial_puppies = 8 →
  stud_fee = 300 →
  profit = 1500 →
  (initial_puppies / 2 - 1) > 0 →
  (profit + stud_fee) / (initial_puppies / 2 - 1) = 600 := by
sorry

end NUMINAMATH_CALUDE_puppy_price_calculation_l1777_177779


namespace NUMINAMATH_CALUDE_expression_simplification_l1777_177796

theorem expression_simplification (b : ℝ) (h : b ≠ -1/2) :
  1 - (1 / (1 + b / (1 + b))) = b / (1 + 2*b) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1777_177796


namespace NUMINAMATH_CALUDE_percentage_problem_l1777_177721

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1777_177721


namespace NUMINAMATH_CALUDE_total_triangles_l1777_177741

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  small_triangles : ℕ

/-- Counts the total number of triangles in a divided triangle -/
def count_triangles (t : DividedTriangle) : ℕ :=
  t.small_triangles + (t.small_triangles - 1) + 1

/-- The problem setup -/
def triangle_problem : Prop :=
  ∃ (t1 t2 : DividedTriangle),
    t1.small_triangles = 3 ∧
    t2.small_triangles = 3 ∧
    count_triangles t1 + count_triangles t2 = 13

/-- The theorem to prove -/
theorem total_triangles : triangle_problem := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_l1777_177741


namespace NUMINAMATH_CALUDE_cube_root_of_product_l1777_177762

theorem cube_root_of_product (a b c : ℕ) : 
  (2^6 * 3^3 * 5^3 : ℝ)^(1/3) = 60 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l1777_177762


namespace NUMINAMATH_CALUDE_susies_golden_comets_l1777_177773

theorem susies_golden_comets (susie_rir : ℕ) (britney_total susie_total : ℕ) : ℕ :=
  let susie_gc := britney_total - susie_total - 8
  have h1 : susie_rir = 11 := by sorry
  have h2 : britney_total = susie_total + 8 := by sorry
  have h3 : britney_total = 2 * susie_rir + susie_gc / 2 := by sorry
  have h4 : susie_total = susie_rir + susie_gc := by sorry
  6

#check susies_golden_comets

end NUMINAMATH_CALUDE_susies_golden_comets_l1777_177773


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_boat_distance_along_stream_proof_l1777_177754

/-- The distance a boat travels along a stream in one hour, given its speed in still water and its distance against the stream in one hour. -/
theorem boat_distance_along_stream 
  (boat_speed : ℝ)  -- Speed of the boat in still water
  (distance_against : ℝ)  -- Distance traveled against the stream in one hour
  (h1 : boat_speed = 9)  -- The boat's speed in still water is 9 km/hr
  (h2 : distance_against = 7)  -- The boat travels 7 km against the stream in one hour
  : ℝ :=  -- The distance the boat travels along the stream in one hour
11

theorem boat_distance_along_stream_proof 
  (boat_speed : ℝ) 
  (distance_against : ℝ) 
  (h1 : boat_speed = 9) 
  (h2 : distance_against = 7) 
  : boat_distance_along_stream boat_speed distance_against h1 h2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_boat_distance_along_stream_proof_l1777_177754


namespace NUMINAMATH_CALUDE_steinburg_marching_band_max_size_l1777_177794

theorem steinburg_marching_band_max_size :
  ∀ n : ℕ,
  (30 * n) % 34 = 6 →
  30 * n < 1200 →
  (∀ m : ℕ, (30 * m) % 34 = 6 → 30 * m < 1200 → 30 * m ≤ 30 * n) →
  30 * n = 720 := by
sorry

end NUMINAMATH_CALUDE_steinburg_marching_band_max_size_l1777_177794


namespace NUMINAMATH_CALUDE_henry_lap_time_l1777_177777

theorem henry_lap_time (margo_lap_time henry_lap_time meet_time : ℕ) 
  (h1 : margo_lap_time = 12)
  (h2 : meet_time = 84)
  (h3 : meet_time % margo_lap_time = 0)
  (h4 : meet_time % henry_lap_time = 0)
  (h5 : henry_lap_time < margo_lap_time)
  : henry_lap_time = 7 :=
sorry

end NUMINAMATH_CALUDE_henry_lap_time_l1777_177777


namespace NUMINAMATH_CALUDE_problem_solution_l1777_177771

-- Given equation
def equation (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0

-- Define the system of equations
def system (a b c x y : ℝ) : Prop :=
  a*x + b*y = 30 ∧ c*x + a*y = 28

-- Define the quadratic equation
def quadratic (a b m x : ℝ) : Prop :=
  a*x^2 + b*x + m = 0

theorem problem_solution :
  ∀ a b c : ℝ, equation a b c →
  (∃ x y : ℝ, (a = 3 ∧ b = 4 ∧ c = 5) → system a b c x y ∧ x = 2 ∧ y = 6) ∧
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) ∧
  (∀ m : ℝ, (∃ x : ℝ, quadratic a b m x) → m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1777_177771


namespace NUMINAMATH_CALUDE_solution_count_l1777_177758

-- Define the function f
def f (n : ℤ) : ℤ := ⌈(149 * n : ℚ) / 150⌉ - ⌊(150 * n : ℚ) / 151⌋

-- State the theorem
theorem solution_count : 
  (∃ (S : Finset ℤ), (∀ n ∈ S, 1 + ⌊(150 * n : ℚ) / 151⌋ = ⌈(149 * n : ℚ) / 150⌉) ∧ S.card = 15150) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l1777_177758


namespace NUMINAMATH_CALUDE_remaining_money_l1777_177790

/-- Calculates the remaining money after spending on sweets and giving to friends -/
theorem remaining_money 
  (initial_amount : ℚ)
  (spent_on_sweets : ℚ)
  (given_to_each_friend : ℚ)
  (number_of_friends : ℕ)
  (h1 : initial_amount = 7.1)
  (h2 : spent_on_sweets = 1.05)
  (h3 : given_to_each_friend = 1)
  (h4 : number_of_friends = 2) :
  initial_amount - spent_on_sweets - (given_to_each_friend * number_of_friends) = 4.05 := by
  sorry

#eval (7.1 : ℚ) - 1.05 - (1 * 2)  -- This should evaluate to 4.05

end NUMINAMATH_CALUDE_remaining_money_l1777_177790


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1777_177756

theorem trigonometric_simplification (α : ℝ) : 
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1777_177756


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l1777_177769

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (sale_discount : ℝ) 
  (profit_margin : ℝ) 
  (h1 : purchase_discount = 0.3) 
  (h2 : sale_discount = 0.2) 
  (h3 : profit_margin = 0.3) 
  (h4 : list_price > 0) :
  let purchase_price := list_price * (1 - purchase_discount)
  let marked_price := list_price * 1.25
  let selling_price := marked_price * (1 - sale_discount)
  selling_price = purchase_price * (1 + profit_margin) := by
sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l1777_177769


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l1777_177722

/-- The number of revolutions needed for the second horse to cover the same distance as the first horse on a merry-go-round -/
theorem merry_go_round_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 = 30) (hr2 : r2 = 10) (hn1 : n1 = 40) :
  let d1 := 2 * Real.pi * r1 * n1
  let n2 := d1 / (2 * Real.pi * r2)
  n2 = 120 := by sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l1777_177722


namespace NUMINAMATH_CALUDE_butterfly_flight_l1777_177742

theorem butterfly_flight (field_length field_width start_distance : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 15)
  (h3 : start_distance = 6)
  (h4 : start_distance < field_length / 2) :
  let end_distance := field_length - 2 * start_distance
  let flight_distance := Real.sqrt (field_width ^ 2 + end_distance ^ 2)
  flight_distance = 17 := by sorry

end NUMINAMATH_CALUDE_butterfly_flight_l1777_177742


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1777_177705

theorem lattice_points_on_hyperbola : 
  ∃! (points : Finset (ℤ × ℤ)), 
    points.card = 4 ∧ 
    ∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1777_177705


namespace NUMINAMATH_CALUDE_article_cost_changes_l1777_177709

theorem article_cost_changes (original_cost : ℝ) : 
  original_cost = 75 → 
  (original_cost * 1.2) * 0.8 = 72 := by
sorry

end NUMINAMATH_CALUDE_article_cost_changes_l1777_177709


namespace NUMINAMATH_CALUDE_min_pencils_for_ten_correct_l1777_177740

/-- Represents the number of pencils of each color in the drawer -/
structure PencilDrawer :=
  (orange : ℕ)
  (purple : ℕ)
  (grey : ℕ)
  (cyan : ℕ)
  (violet : ℕ)

/-- The minimum number of pencils to ensure at least 10 of one color -/
def minPencilsForTen (drawer : PencilDrawer) : ℕ := 43

/-- Theorem stating the minimum number of pencils needed -/
theorem min_pencils_for_ten_correct (drawer : PencilDrawer) 
  (h1 : drawer.orange = 26)
  (h2 : drawer.purple = 22)
  (h3 : drawer.grey = 18)
  (h4 : drawer.cyan = 15)
  (h5 : drawer.violet = 10) :
  minPencilsForTen drawer = 43 ∧
  ∀ n : ℕ, n < 43 → ¬(∃ color : ℕ, color ≥ 10 ∧ 
    (color ≤ drawer.orange ∨ 
     color ≤ drawer.purple ∨ 
     color ≤ drawer.grey ∨ 
     color ≤ drawer.cyan ∨ 
     color ≤ drawer.violet)) := by
  sorry

end NUMINAMATH_CALUDE_min_pencils_for_ten_correct_l1777_177740


namespace NUMINAMATH_CALUDE_two_digit_number_ratio_l1777_177736

theorem two_digit_number_ratio (a b : ℕ) (h1 : 10 * a + b - (10 * b + a) = 36) (h2 : (a + b) - (a - b) = 8) : 
  a = 2 * b := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_ratio_l1777_177736


namespace NUMINAMATH_CALUDE_expected_attempts_proof_l1777_177712

/-- The expected number of attempts to open a safe with n keys -/
def expected_attempts (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / 2

/-- Theorem stating that the expected number of attempts to open a safe
    with n keys distributed sequentially to n students is (n+1)/2 -/
theorem expected_attempts_proof (n : ℕ) :
  expected_attempts n = (n + 1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_attempts_proof_l1777_177712


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1777_177737

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1777_177737


namespace NUMINAMATH_CALUDE_exists_thirteen_cubes_l1777_177788

/-- Represents a 4x4 board with cube stacks -/
def Board := Fin 4 → Fin 4 → ℕ

/-- Predicate to check if the board configuration is valid -/
def valid_board (b : Board) : Prop :=
  ∀ n : Fin 8, ∃! (i j k l : Fin 4), 
    b i j = n + 1 ∧ b k l = n + 1 ∧ (i ≠ k ∨ j ≠ l)

/-- Theorem stating that there exists a pair of cells with 13 cubes total -/
theorem exists_thirteen_cubes (b : Board) (h : valid_board b) : 
  ∃ (i j k l : Fin 4), b i j + b k l = 13 :=
sorry

end NUMINAMATH_CALUDE_exists_thirteen_cubes_l1777_177788


namespace NUMINAMATH_CALUDE_money_distribution_l1777_177720

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 900)
  (AC_sum : A + C = 400)
  (C_amount : C = 250) :
  B + C = 750 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l1777_177720


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1777_177701

theorem sufficient_not_necessary (a : ℝ) :
  (a < -1 → ∃ x₀ : ℝ, a * Real.sin x₀ + 1 < 0) ∧
  (∃ a : ℝ, a ≥ -1 ∧ ∃ x₀ : ℝ, a * Real.sin x₀ + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1777_177701


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1777_177783

theorem system_of_equations_solution (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  3 * a = 9 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1777_177783


namespace NUMINAMATH_CALUDE_class_grouping_l1777_177723

/-- Given a class where students can form 8 pairs when grouped in twos,
    prove that the number of groups formed when students are grouped in fours is 4. -/
theorem class_grouping (num_pairs : ℕ) (h : num_pairs = 8) :
  (2 * num_pairs) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_class_grouping_l1777_177723


namespace NUMINAMATH_CALUDE_product_evaluation_l1777_177735

theorem product_evaluation (a : ℤ) (h : a = 3) :
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1777_177735


namespace NUMINAMATH_CALUDE_train_speeds_l1777_177799

/-- Represents the speeds of two trains and the conditions of their journey --/
structure TrainJourney where
  distance : ℝ
  speed_difference : ℝ
  time_difference : ℝ
  slow_speed : ℝ
  fast_speed : ℝ

/-- Theorem stating the conditions and expected speeds of the trains --/
theorem train_speeds (journey : TrainJourney) 
  (h1 : journey.distance = 650)
  (h2 : journey.speed_difference = 35)
  (h3 : journey.time_difference = 3.5)
  (h4 : journey.fast_speed = journey.slow_speed + journey.speed_difference)
  (h5 : journey.distance / journey.slow_speed - journey.distance / journey.fast_speed = journey.time_difference) :
  journey.slow_speed = 65 ∧ journey.fast_speed = 100 := by
  sorry

#check train_speeds

end NUMINAMATH_CALUDE_train_speeds_l1777_177799


namespace NUMINAMATH_CALUDE_problem_solution_l1777_177753

theorem problem_solution (x y : ℝ) (h : -x + 2*y = 5) :
  5*(x - 2*y)^2 - 3*(x - 2*y) - 60 = 80 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1777_177753


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l1777_177729

theorem min_value_of_expression (x y : ℝ) : 
  (x^2*y - 1)^2 + (x^2 + y)^2 ≥ 1 :=
sorry

theorem min_value_achievable : 
  ∃ x y : ℝ, (x^2*y - 1)^2 + (x^2 + y)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l1777_177729


namespace NUMINAMATH_CALUDE_manager_percentage_reduction_l1777_177782

/-- Calculates the percentage of managers after some leave the room. -/
def target_percentage (total_employees : ℕ) (initial_percentage : ℚ) (managers_leaving : ℚ) : ℚ :=
  let initial_managers : ℚ := (initial_percentage / 100) * total_employees
  let remaining_managers : ℚ := initial_managers - managers_leaving
  (remaining_managers / total_employees) * 100

/-- The target percentage of managers is approximately 49% -/
theorem manager_percentage_reduction :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (target_percentage 100 99 49.99999999999996 - 49) < ε :=
sorry

end NUMINAMATH_CALUDE_manager_percentage_reduction_l1777_177782


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1777_177775

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1777_177775


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l1777_177747

/-- The probability of Sheila attending the picnic -/
def probability_attend : ℝ := 0.55

/-- The probability of rain -/
def p_rain : ℝ := 0.30

/-- The probability of sunny weather -/
def p_sunny : ℝ := 0.50

/-- The probability of partly cloudy weather -/
def p_partly_cloudy : ℝ := 0.20

/-- The probability Sheila attends if it rains -/
def p_attend_rain : ℝ := 0.15

/-- The probability Sheila attends if it's sunny -/
def p_attend_sunny : ℝ := 0.85

/-- The probability Sheila attends if it's partly cloudy -/
def p_attend_partly_cloudy : ℝ := 0.40

/-- Theorem stating that the probability of Sheila attending the picnic is correct -/
theorem sheila_picnic_probability : 
  probability_attend = p_rain * p_attend_rain + p_sunny * p_attend_sunny + p_partly_cloudy * p_attend_partly_cloudy :=
by sorry

end NUMINAMATH_CALUDE_sheila_picnic_probability_l1777_177747


namespace NUMINAMATH_CALUDE_diophantus_age_problem_l1777_177708

theorem diophantus_age_problem :
  ∀ (x : ℕ),
    (x / 6 : ℚ) + (x / 12 : ℚ) + (x / 7 : ℚ) + 5 + (x / 2 : ℚ) + 4 = x →
    x = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_diophantus_age_problem_l1777_177708


namespace NUMINAMATH_CALUDE_basketball_team_age_stats_l1777_177718

/-- Represents the age distribution of players in a basketball team -/
structure AgeDistribution :=
  (age18 : ℕ)
  (age19 : ℕ)
  (age20 : ℕ)
  (age21 : ℕ)
  (total : ℕ)
  (sum : ℕ)
  (h_total : age18 + age19 + age20 + age21 = total)
  (h_sum : 18 * age18 + 19 * age19 + 20 * age20 + 21 * age21 = sum)

/-- The mode of a set of ages -/
def mode (d : AgeDistribution) : ℕ :=
  max (max d.age18 d.age19) (max d.age20 d.age21)

/-- The mean of a set of ages -/
def mean (d : AgeDistribution) : ℚ :=
  d.sum / d.total

/-- Theorem stating the mode and mean of the given age distribution -/
theorem basketball_team_age_stats :
  ∃ d : AgeDistribution,
    d.age18 = 5 ∧
    d.age19 = 4 ∧
    d.age20 = 1 ∧
    d.age21 = 2 ∧
    d.total = 12 ∧
    mode d = 18 ∧
    mean d = 19 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_age_stats_l1777_177718


namespace NUMINAMATH_CALUDE_import_tax_problem_l1777_177780

theorem import_tax_problem (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) (total_value : ℝ) : 
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 109.90 →
  tax_paid = tax_rate * (total_value - tax_threshold) →
  total_value = 2570 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_problem_l1777_177780


namespace NUMINAMATH_CALUDE_roger_retirement_eligibility_l1777_177734

theorem roger_retirement_eligibility 
  (roger peter tom robert mike sarah laura james : ℕ) 
  (h1 : roger = peter + tom + robert + mike + sarah + laura + james)
  (h2 : peter = 12)
  (h3 : tom = 2 * robert)
  (h4 : robert = peter - 4)
  (h5 : robert = mike + 2)
  (h6 : sarah = mike + 3)
  (h7 : sarah = tom / 2)
  (h8 : laura = robert - mike)
  (h9 : james > 0) : 
  roger > 50 := by
  sorry

end NUMINAMATH_CALUDE_roger_retirement_eligibility_l1777_177734


namespace NUMINAMATH_CALUDE_cooks_selection_theorem_l1777_177792

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem cooks_selection_theorem (total_people : ℕ) (cooks_needed : ℕ) (invalid_combinations : ℕ) :
  total_people = 10 →
  cooks_needed = 3 →
  invalid_combinations = choose 8 1 →
  choose total_people cooks_needed - invalid_combinations = 112 := by
sorry

end NUMINAMATH_CALUDE_cooks_selection_theorem_l1777_177792


namespace NUMINAMATH_CALUDE_roots_of_polynomials_product_DE_l1777_177763

theorem roots_of_polynomials (r : ℝ) : 
  r^2 = r + 1 → r^6 = 8*r + 5 := by sorry

theorem product_DE : ∃ (D E : ℤ), 
  (∀ (r : ℝ), r^2 = r + 1 → r^6 = D*r + E) ∧ D*E = 40 := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_product_DE_l1777_177763


namespace NUMINAMATH_CALUDE_triangle_side_length_l1777_177765

theorem triangle_side_length (A B C : ℝ) (cos_half_C BC AC : ℝ) :
  cos_half_C = Real.sqrt 5 / 5 →
  BC = 1 →
  AC = 5 →
  Real.sqrt ((BC^2 + AC^2) - 2 * BC * AC * (2 * cos_half_C^2 - 1)) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1777_177765


namespace NUMINAMATH_CALUDE_broken_bamboo_equation_l1777_177793

theorem broken_bamboo_equation (x : ℝ) : 
  (0 ≤ x) ∧ (x ≤ 10) →
  x^2 + 3^2 = (10 - x)^2 :=
by sorry

/- Explanation of the Lean 4 statement:
   - We import Mathlib to access necessary mathematical definitions and theorems.
   - We define a theorem named 'broken_bamboo_equation'.
   - The theorem takes a real number 'x' as input, representing the height of the broken part.
   - The condition (0 ≤ x) ∧ (x ≤ 10) ensures that x is between 0 and 10 chi.
   - The equation x^2 + 3^2 = (10 - x)^2 represents the Pythagorean theorem applied to the scenario.
   - We use 'by sorry' to skip the proof, as requested.
-/

end NUMINAMATH_CALUDE_broken_bamboo_equation_l1777_177793


namespace NUMINAMATH_CALUDE_apple_purchase_theorem_l1777_177766

/-- The cost of apples with a two-tier pricing system -/
def apple_cost (l q : ℚ) (x : ℚ) : ℚ :=
  if x ≤ 30 then l * x
  else l * 30 + q * (x - 30)

theorem apple_purchase_theorem (l q : ℚ) :
  (∀ x, x ≤ 30 → apple_cost l q x = l * x) ∧
  (∀ x, x > 30 → apple_cost l q x = l * 30 + q * (x - 30)) ∧
  (apple_cost l q 36 = 366) ∧
  (apple_cost l q 15 = 150) ∧
  (∃ x, apple_cost l q x = 333) →
  ∃ x, apple_cost l q x = 333 ∧ x = 33 :=
by sorry

end NUMINAMATH_CALUDE_apple_purchase_theorem_l1777_177766


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1777_177760

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1}

theorem intersection_with_complement : A ∩ (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1777_177760


namespace NUMINAMATH_CALUDE_sum_of_roots_l1777_177745

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 15*a^2 + 20*a - 50 = 0)
  (hb : 8*b^3 - 60*b^2 - 290*b + 2575 = 0) : 
  a + b = 15/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1777_177745


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1777_177731

theorem polynomial_factorization :
  ∀ x : ℝ, x^15 + x^7 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1777_177731


namespace NUMINAMATH_CALUDE_divisibility_of_integer_part_l1777_177704

theorem divisibility_of_integer_part (m : ℕ) 
  (h_odd : m % 2 = 1) 
  (h_not_div_3 : m % 3 ≠ 0) : 
  ∃ k : ℤ, (4^m : ℝ) - (2 + Real.sqrt 2)^m = k + (112 : ℝ) * ↑(⌊((4^m : ℝ) - (2 + Real.sqrt 2)^m) / 112⌋) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_integer_part_l1777_177704


namespace NUMINAMATH_CALUDE_integral_cos_squared_sin_l1777_177787

theorem integral_cos_squared_sin (x : Real) :
  deriv (fun x => -Real.cos x ^ 3 / 3) x = Real.cos x ^ 2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_squared_sin_l1777_177787


namespace NUMINAMATH_CALUDE_class_composition_l1777_177746

/-- Represents the percentage of men in a college class -/
def percentage_men : ℝ := 40

/-- Represents the percentage of women in a college class -/
def percentage_women : ℝ := 100 - percentage_men

/-- Represents the percentage of women who are science majors -/
def women_science_percentage : ℝ := 20

/-- Represents the percentage of non-science majors in the class -/
def non_science_percentage : ℝ := 60

/-- Represents the percentage of men who are science majors -/
def men_science_percentage : ℝ := 70

theorem class_composition :
  percentage_men + percentage_women = 100 ∧
  women_science_percentage / 100 * percentage_women +
    men_science_percentage / 100 * percentage_men = 100 - non_science_percentage :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l1777_177746


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1777_177714

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- Define the interval
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- State the theorem
theorem max_min_values_of_f :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 8) ∧
  (∃ x ∈ interval, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1777_177714


namespace NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_achievable_l1777_177733

theorem min_trig_expression (θ : Real) (h_acute : 0 < θ ∧ θ < π / 2) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) +
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) ≥ 2 :=
by sorry

theorem min_trig_expression_achievable :
  ∃ θ : Real, 0 < θ ∧ θ < π / 2 ∧
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) +
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_achievable_l1777_177733


namespace NUMINAMATH_CALUDE_exam_pass_count_l1777_177743

theorem exam_pass_count (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 35 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed failed : ℕ), 
    passed + failed = total ∧
    passed * passed_avg + failed * failed_avg = total * overall_avg ∧
    passed = 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_count_l1777_177743


namespace NUMINAMATH_CALUDE_range_of_a_l1777_177751

theorem range_of_a (a : ℝ) (ha : a ≠ 0) : 
  let A := {x : ℝ | x^2 - x - 6 < 0}
  let B := {x : ℝ | x^2 + 2*x - 8 ≥ 0}
  let C := {x : ℝ | x^2 - 4*a*x + 3*a^2 < 0}
  C ⊆ (A ∩ (Set.univ \ B)) →
  (0 < a ∧ a ≤ 2/3) ∨ (-2/3 ≤ a ∧ a < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1777_177751


namespace NUMINAMATH_CALUDE_exam_students_count_l1777_177727

/-- The total number of students in an examination -/
def total_students : ℕ := 300

/-- The number of students who just passed -/
def students_just_passed : ℕ := 60

/-- The percentage of students who got first division -/
def first_division_percent : ℚ := 26 / 100

/-- The percentage of students who got second division -/
def second_division_percent : ℚ := 54 / 100

/-- Theorem stating that the total number of students is 300 -/
theorem exam_students_count :
  (students_just_passed : ℚ) / total_students = 1 - first_division_percent - second_division_percent :=
by sorry

end NUMINAMATH_CALUDE_exam_students_count_l1777_177727


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1777_177713

/-- The greatest distance between centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 16)
  (h_height : rectangle_height = 20)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 52 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1777_177713
