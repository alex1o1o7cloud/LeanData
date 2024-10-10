import Mathlib

namespace bob_initial_nickels_l1380_138045

theorem bob_initial_nickels (a b : ℕ) 
  (h1 : b + 1 = 4 * (a - 1)) 
  (h2 : b - 1 = 3 * (a + 1)) : 
  b = 31 := by sorry

end bob_initial_nickels_l1380_138045


namespace sum_of_coefficients_sum_of_coefficients_is_negative_nineteen_l1380_138088

theorem sum_of_coefficients (x : ℝ) : 
  3 * (x^8 - 2*x^5 + x^3 - 7) - 5 * (x^6 + 3*x^2 - 6) + 2 * (x^4 - 5) = 
  3*x^8 - 5*x^6 - 6*x^5 + 2*x^4 + 3*x^3 - 15*x^2 - 1 :=
by sorry

theorem sum_of_coefficients_is_negative_nineteen : 
  (3 * (1^8 - 2*1^5 + 1^3 - 7) - 5 * (1^6 + 3*1^2 - 6) + 2 * (1^4 - 5)) = -19 :=
by sorry

end sum_of_coefficients_sum_of_coefficients_is_negative_nineteen_l1380_138088


namespace angle_sum_is_180_l1380_138092

-- Define angles as real numbers (representing degrees)
variable (angle1 angle2 angle3 : ℝ)

-- Define the properties of vertical angles and supplementary angles
def vertical_angles (a b : ℝ) : Prop := a = b
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

-- State the theorem
theorem angle_sum_is_180 
  (h1 : vertical_angles angle1 angle2) 
  (h2 : supplementary_angles angle2 angle3) : 
  angle1 + angle3 = 180 := by
  sorry

end angle_sum_is_180_l1380_138092


namespace four_links_sufficient_l1380_138084

/-- Represents a chain of links -/
structure Chain :=
  (length : ℕ)
  (link_weight : ℕ)

/-- Represents the ability to create all weights up to the chain's total weight -/
def can_create_all_weights (c : Chain) (separated_links : ℕ) : Prop :=
  ∀ w : ℕ, w ≤ c.length → ∃ (subset : Finset ℕ), 
    subset.card ≤ separated_links ∧ 
    (subset.sum (λ _ => c.link_weight) = w ∨ 
     ∃ (remaining : Finset ℕ), remaining.card + subset.card = c.length ∧ 
       remaining.sum (λ _ => c.link_weight) = w)

/-- The main theorem stating that separating 4 links is sufficient for a chain of 150 links -/
theorem four_links_sufficient (c : Chain) (h1 : c.length = 150) (h2 : c.link_weight = 1) : 
  can_create_all_weights c 4 := by
sorry

end four_links_sufficient_l1380_138084


namespace right_triangle_acute_angles_l1380_138066

theorem right_triangle_acute_angles (α β : ℝ) : 
  α = 30 → -- One acute angle is 30 degrees
  α + β + 90 = 180 → -- Sum of angles in a triangle is 180 degrees, and one angle is right (90 degrees)
  β = 60 := by -- The other acute angle is 60 degrees
sorry

end right_triangle_acute_angles_l1380_138066


namespace cos_shift_equivalence_l1380_138016

open Real

theorem cos_shift_equivalence (x : ℝ) :
  cos (2 * x - π / 6) = sin (2 * (x - π / 6) + π / 2) := by sorry

end cos_shift_equivalence_l1380_138016


namespace quadratic_roots_determine_c_l1380_138053

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := -3 * x^2 + c * x - 8

-- State the theorem
theorem quadratic_roots_determine_c :
  (∀ x : ℝ, f c x < 0 ↔ x < 2 ∨ x > 4) →
  (f c 2 = 0 ∧ f c 4 = 0) →
  c = 18 := by sorry

end quadratic_roots_determine_c_l1380_138053


namespace ellipse_eccentricity_arithmetic_sequence_l1380_138023

/-- 
Given an ellipse with major axis length 2a, minor axis length 2b, and focal length 2c,
where these lengths form an arithmetic sequence, prove that the eccentricity is 3/5.
-/
theorem ellipse_eccentricity_arithmetic_sequence 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : 2 * b = a + c)
  (h_ellipse : b^2 = a^2 - c^2)
  (e : ℝ) 
  (h_eccentricity : e = c / a) :
  e = 3/5 := by
sorry

end ellipse_eccentricity_arithmetic_sequence_l1380_138023


namespace smallest_M_for_inequality_l1380_138008

theorem smallest_M_for_inequality : 
  ∃ M : ℝ, (∀ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧ 
  (∀ M' : ℝ, (∀ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 := by
sorry

end smallest_M_for_inequality_l1380_138008


namespace negation_of_proposition_l1380_138004

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 0 → 2^x + x - 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x + x - 1 < 0) :=
by sorry

end negation_of_proposition_l1380_138004


namespace tutor_schedule_lcm_l1380_138070

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 9) 10 = 90 := by
  sorry

end tutor_schedule_lcm_l1380_138070


namespace monotonic_absolute_value_function_l1380_138083

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem monotonic_absolute_value_function (a : ℝ) :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f a x ≤ f a y ∨ f a x ≥ f a y) →
  a ≥ -1 :=
by sorry

end monotonic_absolute_value_function_l1380_138083


namespace simplify_expression_l1380_138019

/-- Proves that the simplified expression (√3 - 1)^(1 - √2) / (√3 + 1)^(1 + √2) equals 2^(1-√2)(4 - 2√3) -/
theorem simplify_expression :
  let x := (Real.sqrt 3 - 1)^(1 - Real.sqrt 2) / (Real.sqrt 3 + 1)^(1 + Real.sqrt 2)
  let y := 2^(1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3)
  x = y := by sorry

end simplify_expression_l1380_138019


namespace gcd_153_119_l1380_138036

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l1380_138036


namespace max_value_of_f_l1380_138017

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_of_f (h : a ≠ 0) :
  (a > 0 → ∃ (M : ℝ), M = 4 * a * Real.exp (-2) ∧ ∀ x, f a x ≤ M) ∧
  (a < 0 → ∃ (M : ℝ), M = 0 ∧ ∀ x, f a x ≤ M) :=
sorry

end

end max_value_of_f_l1380_138017


namespace shaded_area_theorem_l1380_138050

/-- The area of the upper triangle formed by a diagonal line in a 20cm x 15cm rectangle, 
    where the diagonal starts from the corner of a 5cm x 5cm square within the rectangle. -/
theorem shaded_area_theorem (total_width total_height small_square_side : ℝ) 
  (hw : total_width = 20)
  (hh : total_height = 15)
  (hs : small_square_side = 5) : 
  let large_width := total_width - small_square_side
  let large_height := total_height
  let diagonal_slope := large_height / total_width
  let intersection_y := diagonal_slope * small_square_side
  let triangle_base := large_width
  let triangle_height := large_height - intersection_y
  triangle_base * triangle_height / 2 = 84.375 := by
  sorry

#eval (20 - 5) * (15 - 15 / 20 * 5) / 2

end shaded_area_theorem_l1380_138050


namespace band_total_earnings_l1380_138091

/-- Calculates the total earnings of a band given the number of members, 
    earnings per member per gig, and number of gigs played. -/
def bandEarnings (members : ℕ) (earningsPerMember : ℕ) (gigs : ℕ) : ℕ :=
  members * earningsPerMember * gigs

/-- Theorem stating that a band with 4 members, each earning $20 per gig, 
    and having played 5 gigs, earns a total of $400. -/
theorem band_total_earnings : 
  bandEarnings 4 20 5 = 400 := by
  sorry

end band_total_earnings_l1380_138091


namespace distance_AB_is_correct_l1380_138010

/-- The distance between two points A and B, where two people start simultaneously
    and move towards each other under specific conditions. -/
def distance_AB : ℝ :=
  let speed_A : ℝ := 12.5 -- km/h
  let speed_B : ℝ := 10   -- km/h
  let time_to_bank : ℝ := 0.5 -- hours
  let time_to_return : ℝ := 0.5 -- hours
  let time_to_find_card : ℝ := 0.5 -- hours
  let remaining_time : ℝ := 0.25 -- hours
  62.5 -- km

theorem distance_AB_is_correct :
  let speed_A : ℝ := 12.5 -- km/h
  let speed_B : ℝ := 10   -- km/h
  let time_to_bank : ℝ := 0.5 -- hours
  let time_to_return : ℝ := 0.5 -- hours
  let time_to_find_card : ℝ := 0.5 -- hours
  let remaining_time : ℝ := 0.25 -- hours
  distance_AB = 62.5 := by
  sorry

#check distance_AB_is_correct

end distance_AB_is_correct_l1380_138010


namespace environmental_group_allocation_l1380_138014

theorem environmental_group_allocation :
  let total_members : ℕ := 8
  let num_locations : ℕ := 3
  let min_per_location : ℕ := 2

  let allocation_schemes : ℕ := 
    (Nat.choose total_members 2 * Nat.choose 6 2 * Nat.choose 4 4 / 2 +
     Nat.choose total_members 3 * Nat.choose 5 3 * Nat.choose 2 2 / 2) * 
    (Nat.factorial num_locations)

  allocation_schemes = 2940 :=
by sorry

end environmental_group_allocation_l1380_138014


namespace sock_pairs_count_l1380_138063

def white_socks : ℕ := 5
def brown_socks : ℕ := 4
def blue_socks : ℕ := 2
def red_socks : ℕ := 1

def total_socks : ℕ := white_socks + brown_socks + blue_socks + red_socks

theorem sock_pairs_count :
  (blue_socks * white_socks) + (blue_socks * brown_socks) + (blue_socks * red_socks) = 20 := by
  sorry

end sock_pairs_count_l1380_138063


namespace vector_equation_and_parallelism_l1380_138018

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_equation_and_parallelism :
  (a = (5/9 : ℝ) • b + (8/9 : ℝ) • c) ∧
  (∃ (t : ℝ), t • (a + (-16/13 : ℝ) • c) = 2 • b - a) :=
by sorry

end vector_equation_and_parallelism_l1380_138018


namespace pie_eating_contest_l1380_138072

theorem pie_eating_contest (first_student third_student : ℚ) : 
  first_student = 7/8 → third_student = 3/4 → first_student - third_student = 1/8 := by
  sorry

end pie_eating_contest_l1380_138072


namespace gold_coin_problem_l1380_138030

theorem gold_coin_problem (n : ℕ) (c : ℕ) : 
  (n = 10 * (c - 4)) →
  (n = 7 * c + 5) →
  n = 110 := by
sorry

end gold_coin_problem_l1380_138030


namespace edwards_initial_money_l1380_138026

theorem edwards_initial_money :
  ∀ (initial_book_cost : ℝ) (discount_rate : ℝ) (num_books : ℕ) (pen_cost : ℝ) (num_pens : ℕ) (money_left : ℝ),
    initial_book_cost = 40 →
    discount_rate = 0.25 →
    num_books = 100 →
    pen_cost = 2 →
    num_pens = 3 →
    money_left = 6 →
    ∃ (initial_money : ℝ),
      initial_money = initial_book_cost * (1 - discount_rate) + (pen_cost * num_pens) + money_left ∧
      initial_money = 42 :=
by sorry

end edwards_initial_money_l1380_138026


namespace dogs_can_prevent_escape_l1380_138097

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square field -/
structure SquareField where
  sideLength : ℝ
  center : Point
  corners : Fin 4 → Point

/-- Represents the game setup -/
structure WolfDogGame where
  field : SquareField
  wolfSpeed : ℝ
  dogSpeed : ℝ

/-- Checks if a point is on the perimeter of the square field -/
def isOnPerimeter (field : SquareField) (p : Point) : Prop :=
  let a := field.sideLength / 2
  (p.x = field.center.x - a ∨ p.x = field.center.x + a) ∧
  (p.y ≥ field.center.y - a ∧ p.y ≤ field.center.y + a) ∨
  (p.y = field.center.y - a ∨ p.y = field.center.y + a) ∧
  (p.x ≥ field.center.x - a ∧ p.x ≤ field.center.x + a)

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_escape (game : WolfDogGame) 
  (h1 : game.field.sideLength > 0)
  (h2 : game.dogSpeed = 1.5 * game.wolfSpeed)
  (h3 : game.wolfSpeed > 0) :
  ∀ (p : Point), isOnPerimeter game.field p →
    ∃ (t : ℝ), t ≥ 0 ∧ 
      (∃ (i : Fin 4), (t * game.dogSpeed)^2 ≥ 
        ((game.field.corners i).x - p.x)^2 + ((game.field.corners i).y - p.y)^2) ∧
      (t * game.wolfSpeed)^2 < 
        (game.field.center.x - p.x)^2 + (game.field.center.y - p.y)^2 :=
sorry

end dogs_can_prevent_escape_l1380_138097


namespace room_width_l1380_138013

/-- Given a rectangular room with the specified length and paving cost, prove its width. -/
theorem room_width (length : ℝ) (total_cost : ℝ) (rate : ℝ) (width : ℝ) 
  (h1 : length = 5.5)
  (h2 : total_cost = 20625)
  (h3 : rate = 1000)
  (h4 : total_cost = rate * length * width) :
  width = 3.75 := by sorry

end room_width_l1380_138013


namespace trig_identity_l1380_138074

theorem trig_identity (α : Real) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) + Real.sin (α - π / 6) ^ 2 = (2 - Real.sqrt 3) / 3 := by
  sorry

end trig_identity_l1380_138074


namespace monochromatic_right_triangle_exists_l1380_138089

-- Define the color type
inductive Color
| Black
| White

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point
  is_equilateral : sorry -- Condition that ABC is equilateral

-- Define the set G of points on the sides of the triangle
def G (t : EquilateralTriangle) : Set Point :=
  sorry -- Definition of G as described in the problem

-- Define a coloring function
def coloring (p : Point) : Color :=
  sorry -- Some function that assigns Black or White to each point

-- Define what it means for a triangle to be inscribed and right-angled
def is_inscribed_right_triangle (t : EquilateralTriangle) (p q r : Point) : Prop :=
  sorry -- Condition for p, q, r to form an inscribed right triangle in t

-- The main theorem
theorem monochromatic_right_triangle_exists (t : EquilateralTriangle) :
  ∃ p q r : Point, p ∈ G t ∧ q ∈ G t ∧ r ∈ G t ∧
  is_inscribed_right_triangle t p q r ∧
  coloring p = coloring q ∧ coloring q = coloring r :=
sorry

end monochromatic_right_triangle_exists_l1380_138089


namespace largest_prime_divisor_check_l1380_138040

theorem largest_prime_divisor_check (n : ℕ) : 
  1200 ≤ n ∧ n ≤ 1250 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime := by
  sorry

end largest_prime_divisor_check_l1380_138040


namespace impossibleConsecutive_l1380_138065

/-- A move that replaces one number with the sum of both numbers -/
def move (a b : ℕ) : ℕ × ℕ := (a + b, b)

/-- The sequence of numbers obtained after applying moves -/
def boardSequence : ℕ → ℕ × ℕ
  | 0 => (2, 5)
  | n + 1 => let (a, b) := boardSequence n; move a b

/-- The difference between the two numbers on the board after n moves -/
def difference (n : ℕ) : ℕ :=
  let (a, b) := boardSequence n
  max a b - min a b

theorem impossibleConsecutive : ∀ n : ℕ, difference n ≠ 1 := by
  sorry

end impossibleConsecutive_l1380_138065


namespace max_product_of_three_l1380_138061

def S : Set Int := {-9, -5, -3, 0, 2, 6, 8}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a * b * c ≤ 360 :=
sorry

end max_product_of_three_l1380_138061


namespace friend_c_spent_26_l1380_138059

/-- Friend C's lunch cost given the conditions of the problem -/
def friend_c_cost (your_cost friend_a_extra friend_b_less : ℕ) : ℕ :=
  2 * (your_cost + friend_a_extra - friend_b_less)

/-- Theorem stating that Friend C's lunch cost is $26 -/
theorem friend_c_spent_26 : friend_c_cost 12 4 3 = 26 := by
  sorry

end friend_c_spent_26_l1380_138059


namespace markup_rate_calculation_l1380_138021

/-- Represents the markup rate calculation for a product with given profit and expense percentages. -/
theorem markup_rate_calculation (profit_percent : ℝ) (expense_percent : ℝ) :
  profit_percent = 0.12 →
  expense_percent = 0.18 →
  let cost_percent := 1 - profit_percent - expense_percent
  let markup_rate := (1 / cost_percent - 1) * 100
  ∃ ε > 0, abs (markup_rate - 42.857) < ε :=
by
  sorry

end markup_rate_calculation_l1380_138021


namespace remainder_problem_l1380_138034

theorem remainder_problem (x : ℤ) : 
  x % 62 = 7 → (x + 11) % 31 = 18 := by
sorry

end remainder_problem_l1380_138034


namespace rain_forest_animals_l1380_138022

theorem rain_forest_animals (reptile_house : ℕ) (rain_forest : ℕ) : 
  reptile_house = 16 → 
  reptile_house = 3 * rain_forest - 5 → 
  rain_forest = 7 := by
sorry

end rain_forest_animals_l1380_138022


namespace only_f₂_is_saturated_l1380_138042

/-- Definition of a "saturated function of 1" -/
def is_saturated_function_of_1 (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

/-- Function f₁(x) = 1/x -/
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x

/-- Function f₂(x) = 2^x -/
noncomputable def f₂ (x : ℝ) : ℝ := 2^x

/-- Function f₃(x) = log(x² + 2) -/
noncomputable def f₃ (x : ℝ) : ℝ := Real.log (x^2 + 2)

/-- Theorem stating that only f₂ is a "saturated function of 1" -/
theorem only_f₂_is_saturated :
  ¬ is_saturated_function_of_1 f₁ ∧
  is_saturated_function_of_1 f₂ ∧
  ¬ is_saturated_function_of_1 f₃ :=
sorry

end only_f₂_is_saturated_l1380_138042


namespace quadratic_equation_roots_l1380_138025

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x * (x - 2) = x - 2 → x = r₁ ∨ x = r₂) :=
by
  sorry

end quadratic_equation_roots_l1380_138025


namespace stack_height_probability_l1380_138038

/-- Represents the possible heights of a crate -/
inductive CrateHeight : Type
  | Two : CrateHeight
  | Three : CrateHeight
  | Five : CrateHeight

/-- The number of crates in the stack -/
def numCrates : ℕ := 5

/-- The target height of the stack -/
def targetHeight : ℕ := 16

/-- Calculates the total number of possible arrangements -/
def totalArrangements : ℕ := 3^numCrates

/-- Calculates the number of valid arrangements that sum to the target height -/
def validArrangements : ℕ := 20

/-- The probability of achieving the target height -/
def probabilityTargetHeight : ℚ := validArrangements / totalArrangements

theorem stack_height_probability :
  probabilityTargetHeight = 20 / 243 := by sorry

end stack_height_probability_l1380_138038


namespace firm_employs_100_looms_l1380_138093

/-- Represents the number of looms employed by the textile manufacturing firm. -/
def number_of_looms : ℕ := sorry

/-- The aggregate sales value of the output of the looms in rupees. -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees. -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees. -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for one month, in rupees. -/
def profit_decrease : ℕ := 3500

theorem firm_employs_100_looms :
  number_of_looms = 100 ∧
  aggregate_sales / number_of_looms - manufacturing_expenses / number_of_looms = profit_decrease :=
sorry

end firm_employs_100_looms_l1380_138093


namespace product_remainder_l1380_138095

theorem product_remainder (a b c : ℕ) (h : a = 1625 ∧ b = 1627 ∧ c = 1629) : 
  (a * b * c) % 12 = 3 := by
sorry

end product_remainder_l1380_138095


namespace right_triangle_medians_l1380_138047

theorem right_triangle_medians (D E F : ℝ × ℝ) : 
  -- D is the right angle
  (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0 →
  -- Length of median from D to midpoint of EF is 3√5
  ((E.1 + F.1) / 2 - D.1)^2 + ((E.2 + F.2) / 2 - D.2)^2 = 45 →
  -- Length of median from E to midpoint of DF is 5
  ((D.1 + F.1) / 2 - E.1)^2 + ((D.2 + F.2) / 2 - E.2)^2 = 25 →
  -- Then the length of DE is 2√14
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = 56 :=
by sorry

end right_triangle_medians_l1380_138047


namespace sum_of_digits_2003n_l1380_138090

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sum_of_digits_2003n (n : ℕ) 
  (h_pos : n > 0)
  (h_sum_n : sum_of_digits n = 111)
  (h_sum_7002n : sum_of_digits (7002 * n) = 990) : 
  sum_of_digits (2003 * n) = 555 := by
  sorry

end sum_of_digits_2003n_l1380_138090


namespace inequality_system_solution_l1380_138035

/-- The system of inequalities:
    11x² + 8xy + 8y² ≤ 3
    x - 4y ≤ -3
    has the solution (-1/3, 2/3) -/
theorem inequality_system_solution :
  let x : ℚ := -1/3
  let y : ℚ := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧
  x - 4 * y ≤ -3 := by
  sorry

#check inequality_system_solution

end inequality_system_solution_l1380_138035


namespace inscribed_circle_radius_in_one_third_sector_l1380_138081

/-- The radius of a circle inscribed in a sector that is one-third of a circle with radius 5 cm -/
theorem inscribed_circle_radius_in_one_third_sector :
  ∃ (r : ℝ), 
    r > 0 ∧ 
    r * (Real.sqrt 3 + 1) = 5 ∧
    r = (5 * Real.sqrt 3 - 5) / 2 := by
  sorry

end inscribed_circle_radius_in_one_third_sector_l1380_138081


namespace no_mem_is_veen_l1380_138003

-- Define the universe of discourse
variable {U : Type}

-- Define predicates for Mem, En, and Veen
variable (Mem En Veen : U → Prop)

-- Theorem statement
theorem no_mem_is_veen 
  (h1 : ∀ x, Mem x → En x)  -- All Mems are Ens
  (h2 : ∀ x, En x → ¬Veen x)  -- No Ens are Veens
  : ∀ x, Mem x → ¬Veen x :=  -- No Mem is a Veen
by
  sorry  -- Proof is omitted as per instructions

end no_mem_is_veen_l1380_138003


namespace exponent_multiplication_l1380_138048

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l1380_138048


namespace greatest_prime_factor_of_247_l1380_138037

theorem greatest_prime_factor_of_247 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 247 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 247 → q ≤ p :=
sorry

end greatest_prime_factor_of_247_l1380_138037


namespace equation_has_four_solutions_l1380_138005

/-- The number of integer solutions to the equation 6y^2 + 3xy + x + 2y - 72 = 0 -/
def num_solutions : ℕ := 4

/-- The equation 6y^2 + 3xy + x + 2y - 72 = 0 -/
def equation (x y : ℤ) : Prop :=
  6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0

theorem equation_has_four_solutions :
  ∃ (s : Finset (ℤ × ℤ)), s.card = num_solutions ∧
  (∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2) :=
sorry

end equation_has_four_solutions_l1380_138005


namespace minimum_games_for_95_percent_win_rate_l1380_138099

theorem minimum_games_for_95_percent_win_rate : 
  ∃ N : ℕ, (N = 37 ∧ (1 + N : ℚ) / (3 + N) ≥ 95 / 100) ∧
  ∀ M : ℕ, M < N → (1 + M : ℚ) / (3 + M) < 95 / 100 := by
  sorry

end minimum_games_for_95_percent_win_rate_l1380_138099


namespace team_games_count_l1380_138060

/-- Proves that a team playing under specific win conditions played 120 games in total -/
theorem team_games_count (first_games : ℕ) (first_win_rate : ℚ) (remaining_win_rate : ℚ) (total_win_rate : ℚ) : 
  first_games = 30 →
  first_win_rate = 2/5 →
  remaining_win_rate = 4/5 →
  total_win_rate = 7/10 →
  ∃ (total_games : ℕ), 
    total_games = 120 ∧
    (first_win_rate * first_games + remaining_win_rate * (total_games - first_games) : ℚ) = total_win_rate * total_games :=
by sorry

end team_games_count_l1380_138060


namespace first_player_winning_strategy_l1380_138044

/-- Represents the state of the game with two piles of candies -/
structure GameState where
  p : Nat
  q : Nat

/-- Determines if a given number is a winning number (congruent to 0, 1, or 4 mod 5) -/
def isWinningNumber (n : Nat) : Prop :=
  n % 5 = 0 ∨ n % 5 = 1 ∨ n % 5 = 4

/-- Determines if a given game state is a winning state for the first player -/
def isWinningState (state : GameState) : Prop :=
  isWinningNumber state.p ∨ isWinningNumber state.q

/-- Theorem stating the winning condition for the first player -/
theorem first_player_winning_strategy (state : GameState) :
  (∃ (strategy : GameState → GameState), 
    (∀ (opponent_move : GameState → GameState), 
      strategy (opponent_move (strategy state)) = state)) ↔ 
  isWinningState state :=
sorry

end first_player_winning_strategy_l1380_138044


namespace x_eighteenth_equals_negative_one_l1380_138075

theorem x_eighteenth_equals_negative_one (x : ℂ) (h : x + 1/x = Real.sqrt 3) : x^18 = -1 := by
  sorry

end x_eighteenth_equals_negative_one_l1380_138075


namespace transport_cost_effectiveness_l1380_138011

/-- Represents the transportation cost functions and conditions for fruit shipping --/
structure FruitTransport where
  x : ℝ  -- distance in kilometers
  truck_cost : ℝ → ℝ  -- trucking company cost function
  train_cost : ℝ → ℝ  -- train freight station cost function

/-- Theorem stating the cost-effectiveness of different transportation methods --/
theorem transport_cost_effectiveness (ft : FruitTransport) 
  (h_truck : ft.truck_cost = λ x => 94 * x + 4000)
  (h_train : ft.train_cost = λ x => 81 * x + 6600) :
  (∀ x, x > 0 ∧ x < 200 → ft.truck_cost x < ft.train_cost x) ∧
  (∀ x, x > 200 → ft.train_cost x < ft.truck_cost x) := by
  sorry

#check transport_cost_effectiveness

end transport_cost_effectiveness_l1380_138011


namespace solution_s_l1380_138068

theorem solution_s (s : ℝ) : 
  Real.sqrt (3 * Real.sqrt (s - 3)) = (9 - s) ^ (1/4) → s = 3.6 := by
  sorry

end solution_s_l1380_138068


namespace shekar_average_marks_l1380_138094

def shekar_scores : List ℕ := [76, 65, 82, 62, 85]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℚ) = 74 := by
  sorry

end shekar_average_marks_l1380_138094


namespace cookie_difference_l1380_138001

/-- Given that Alyssa has 129 cookies and Aiyanna has 140 cookies,
    prove that Aiyanna has 11 more cookies than Alyssa. -/
theorem cookie_difference (alyssa_cookies : ℕ) (aiyanna_cookies : ℕ)
    (h1 : alyssa_cookies = 129)
    (h2 : aiyanna_cookies = 140) :
    aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end cookie_difference_l1380_138001


namespace coffee_beans_remaining_l1380_138052

theorem coffee_beans_remaining (J B B_remaining : ℝ) 
  (h1 : J = 0.25 * (J + B))
  (h2 : J + B_remaining = 0.60 * (J + B))
  (h3 : J > 0)
  (h4 : B > 0) :
  B_remaining / B = 7 / 15 := by
sorry

end coffee_beans_remaining_l1380_138052


namespace pizza_slices_per_friend_l1380_138085

theorem pizza_slices_per_friend (num_friends : ℕ) (total_slices : ℕ) (h1 : num_friends = 4) (h2 : total_slices = 16) :
  ∃ (slices_per_friend : ℕ),
    slices_per_friend * num_friends = total_slices ∧
    slices_per_friend = 4 := by
  sorry

end pizza_slices_per_friend_l1380_138085


namespace tissue_usage_l1380_138033

theorem tissue_usage (initial_tissues : ℕ) (remaining_tissues : ℕ) 
  (alice_usage : ℕ) (bob_multiplier : ℕ) (eve_reduction : ℕ) : 
  initial_tissues = 97 →
  remaining_tissues = 47 →
  alice_usage = 12 →
  bob_multiplier = 2 →
  eve_reduction = 3 →
  initial_tissues - remaining_tissues + 
  alice_usage + bob_multiplier * alice_usage + (alice_usage - eve_reduction) = 95 :=
by sorry

end tissue_usage_l1380_138033


namespace chicago_bulls_wins_conditions_satisfied_l1380_138043

/-- The number of games won by the Chicago Bulls -/
def bulls_wins : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_wins : ℕ := bulls_wins + 5

/-- The total number of games won by both teams -/
def total_wins : ℕ := 145

/-- Theorem stating that the Chicago Bulls won 70 games -/
theorem chicago_bulls_wins : bulls_wins = 70 := by sorry

/-- Theorem proving the conditions are satisfied -/
theorem conditions_satisfied :
  (heat_wins = bulls_wins + 5) ∧ (bulls_wins + heat_wins = total_wins) := by sorry

end chicago_bulls_wins_conditions_satisfied_l1380_138043


namespace sequence_divisibility_and_conditions_l1380_138057

def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem sequence_divisibility_and_conditions :
  (∀ n : ℕ, (15^3 : ℤ) ∣ a n) ∧
  (∀ n : ℕ, (1991 : ℤ) ∣ a n ∧ (1991 : ℤ) ∣ a (n + 1) ∧ (1991 : ℤ) ∣ a (n + 2) ↔ ∃ k : ℕ, n = 89595 * k) :=
sorry

end sequence_divisibility_and_conditions_l1380_138057


namespace lcm_of_462_and_150_l1380_138079

theorem lcm_of_462_and_150 :
  let a : ℕ := 462
  let b : ℕ := 150
  let hcf : ℕ := 30
  Nat.lcm a b = 2310 :=
by
  sorry

end lcm_of_462_and_150_l1380_138079


namespace complex_on_ellipse_real_fraction_l1380_138032

theorem complex_on_ellipse_real_fraction (z : ℂ) :
  let x : ℝ := z.re
  let y : ℝ := z.im
  (x^2 / 9 + y^2 / 16 = 1) →
  ((z - (1 + I)) / (z - I)).im = 0 →
  z = Complex.mk ((3 * Real.sqrt 15) / 4) 1 ∨
  z = Complex.mk (-(3 * Real.sqrt 15) / 4) 1 :=
by sorry

end complex_on_ellipse_real_fraction_l1380_138032


namespace solution_implies_m_value_l1380_138009

theorem solution_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + 12*x - m^2 = 0 ∧ x = 2) → m = 2 ∨ m = -2 := by
  sorry

end solution_implies_m_value_l1380_138009


namespace hyperbola_foci_distance_l1380_138000

/-- The distance between the foci of a hyperbola defined by xy = 2 is 4. -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 2 → 
      (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2)) = 
      (Real.sqrt ((x + f₁.1)^2 + (y + f₁.2)^2) + Real.sqrt ((x + f₂.1)^2 + (y + f₂.2)^2))) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 4 :=
by
  sorry


end hyperbola_foci_distance_l1380_138000


namespace quadratic_shift_l1380_138086

/-- Represents a quadratic function of the form y = (x - h)² + k --/
structure QuadraticFunction where
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function horizontally --/
def shift_horizontal (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { h := f.h - d, k := f.k }

/-- Shifts a quadratic function vertically --/
def shift_vertical (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { h := f.h, k := f.k - d }

/-- The main theorem stating that shifting y = (x + 1)² + 3 by 2 units right and 1 unit down
    results in y = (x - 1)² + 2 --/
theorem quadratic_shift :
  let f := QuadraticFunction.mk (-1) 3
  let g := shift_vertical (shift_horizontal f 2) 1
  g = QuadraticFunction.mk 1 2 := by sorry

end quadratic_shift_l1380_138086


namespace factor_implies_c_value_l1380_138073

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (x + 5) ∣ (c * x^3 + 23 * x^2 - 5 * c * x + 55)) → c = 6.3 := by
sorry

end factor_implies_c_value_l1380_138073


namespace sum_of_coefficients_l1380_138055

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by sorry

end sum_of_coefficients_l1380_138055


namespace solution_set_of_inequality_l1380_138087

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x > 0) ↔ (x < 0 ∨ x > 1/2) :=
by sorry

end solution_set_of_inequality_l1380_138087


namespace min_socks_for_five_correct_min_socks_for_five_optimal_l1380_138056

/-- Represents the colors of socks --/
inductive Color
  | Red
  | White
  | Blue

/-- Represents a drawer of socks --/
structure SockDrawer where
  red : ℕ
  white : ℕ
  blue : ℕ
  red_min : red ≥ 5
  white_min : white ≥ 5
  blue_min : blue ≥ 5

/-- The minimum number of socks to guarantee 5 of the same color --/
def minSocksForFive (drawer : SockDrawer) : ℕ := 13

theorem min_socks_for_five_correct (drawer : SockDrawer) :
  ∀ n : ℕ, n < minSocksForFive drawer →
    ∃ (r w b : ℕ), r < 5 ∧ w < 5 ∧ b < 5 ∧ r + w + b = n :=
  sorry

theorem min_socks_for_five_optimal (drawer : SockDrawer) :
  ∃ (r w b : ℕ), (r = 5 ∨ w = 5 ∨ b = 5) ∧ r + w + b = minSocksForFive drawer :=
  sorry

end min_socks_for_five_correct_min_socks_for_five_optimal_l1380_138056


namespace part1_part2_l1380_138002

-- Define the function f(x) = x / (e^x)
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Define the function g(x) = f(x) - m
noncomputable def g (x m : ℝ) : ℝ := f x - m

-- Theorem for part 1
theorem part1 (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧
   ∀ x, x > 0 → g x m = 0 → x = x₁ ∨ x = x₂) →
  0 < m ∧ m < 1 / Real.exp 1 :=
sorry

-- Theorem for part 2
theorem part2 (a : ℝ) :
  (∃! n : ℤ, (f n)^2 - a * f n > 0 ∧ ∀ x : ℝ, x > 0 → (f x)^2 - a * f x > 0 → ⌊x⌋ = n) →
  2 / Real.exp 2 ≤ a ∧ a < 1 / Real.exp 1 :=
sorry

end part1_part2_l1380_138002


namespace original_average_l1380_138051

theorem original_average (n : ℕ) (A : ℚ) (h1 : n = 15) (h2 : (n : ℚ) * (A + 15) = n * 55) : A = 40 := by
  sorry

end original_average_l1380_138051


namespace percentage_difference_l1380_138054

theorem percentage_difference : 
  (0.12 * 24.2) - (0.10 * 14.2) = 1.484 := by
  sorry

end percentage_difference_l1380_138054


namespace constant_distance_between_bikers_l1380_138049

def distance_between_bikers (t : ℝ) (initial_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) : ℝ :=
  initial_distance + speed_b * t - speed_a * t

theorem constant_distance_between_bikers
  (speed_a : ℝ)
  (speed_b : ℝ)
  (initial_distance : ℝ)
  (h1 : speed_a = 350 / 7)
  (h2 : speed_b = 500 / 10)
  (h3 : initial_distance = 75)
  (t : ℝ) :
  distance_between_bikers t initial_distance speed_a speed_b = initial_distance :=
by sorry

end constant_distance_between_bikers_l1380_138049


namespace dessert_eating_contest_l1380_138031

theorem dessert_eating_contest (student1_pie : ℚ) (student2_pie : ℚ) (student3_cake : ℚ) :
  student1_pie = 5/6 ∧ student2_pie = 7/8 ∧ student3_cake = 1/2 →
  max student1_pie student2_pie - student3_cake = 1/3 :=
by sorry

end dessert_eating_contest_l1380_138031


namespace sin_cos_equation_solutions_l1380_138064

/-- The number of solutions to sin(π/4 * sin x) = cos(π/4 * cos x) in [0, 2π] -/
theorem sin_cos_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ 
    Real.sin (π/4 * Real.sin x) = Real.cos (π/4 * Real.cos x)) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2 * π ∧ 
    Real.sin (π/4 * Real.sin y) = Real.cos (π/4 * Real.cos y) → y ∈ s) :=
by sorry

end sin_cos_equation_solutions_l1380_138064


namespace smallest_four_digit_solution_l1380_138024

def is_valid (x : ℕ) : Prop :=
  (3 * x) % 12 = 6 ∧
  (5 * x + 20) % 15 = 25 ∧
  (3 * x - 2) % 35 = (2 * x) % 35

def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x ≤ 9999

theorem smallest_four_digit_solution :
  is_valid 1274 ∧ is_four_digit 1274 ∧
  ∀ y : ℕ, (is_valid y ∧ is_four_digit y) → 1274 ≤ y :=
by sorry

end smallest_four_digit_solution_l1380_138024


namespace scarf_problem_l1380_138077

theorem scarf_problem (initial_scarves : ℕ) (num_girls : ℕ) (final_scarves : ℕ) : 
  initial_scarves = 20 →
  num_girls = 17 →
  final_scarves ≠ 10 :=
by
  intro h_initial h_girls
  sorry


end scarf_problem_l1380_138077


namespace sanity_indeterminable_likely_vampire_l1380_138015

-- Define the types of beings
inductive Being
| Human
| Vampire

-- Define the mental state
inductive MentalState
| Sane
| Insane

-- Define the claim
def claimsLostMind (b : Being) : Prop := true

-- Theorem 1: It's impossible to determine sanity from the claim
theorem sanity_indeterminable (b : Being) (claim : claimsLostMind b) :
  ¬ ∃ (state : MentalState), (b = Being.Human → state = MentalState.Sane) ∧
                             (b = Being.Vampire → state = MentalState.Sane) :=
sorry

-- Theorem 2: The being is most likely a vampire
theorem likely_vampire (b : Being) (claim : claimsLostMind b) :
  b = Being.Vampire :=
sorry

end sanity_indeterminable_likely_vampire_l1380_138015


namespace special_number_fraction_l1380_138027

theorem special_number_fraction (list : List ℝ) (n : ℝ) :
  list.length = 21 ∧
  n ∉ list ∧
  n = 5 * (list.sum / list.length) →
  n / (list.sum + n) = 1 / 5 := by
sorry

end special_number_fraction_l1380_138027


namespace study_group_composition_l1380_138020

def number_of_selections (n m : ℕ) : ℕ :=
  (Nat.choose n 2) * (Nat.choose m 1) * 6

theorem study_group_composition :
  ∃ (n m : ℕ),
    n + m = 8 ∧
    number_of_selections n m = 90 ∧
    n = 3 ∧
    m = 5 := by
  sorry

end study_group_composition_l1380_138020


namespace total_pencils_eq_twelve_l1380_138080

/-- The number of rows of pencils -/
def num_rows : ℕ := 3

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 4

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem total_pencils_eq_twelve : total_pencils = 12 := by
  sorry

end total_pencils_eq_twelve_l1380_138080


namespace fractional_equation_solution_l1380_138006

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ (2 / x = 1 / (x + 1)) ∧ x = -2 := by sorry

end fractional_equation_solution_l1380_138006


namespace min_sum_squares_l1380_138028

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
    (h_sum : 3 * y₁ + 2 * y₂ + y₃ = 30) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 450/7 ∧ ∃ y₁' y₂' y₃', y₁'^2 + y₂'^2 + y₃'^2 = 450/7 ∧ 
    y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 3 * y₁' + 2 * y₂' + y₃' = 30 :=
by sorry


end min_sum_squares_l1380_138028


namespace intersection_of_E_l1380_138071

def E (k : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ |p.1|^k ∧ |p.1| ≥ 1}

theorem intersection_of_E :
  (⋂ k ∈ Finset.range 1991, E (k + 1)) = {p : ℝ × ℝ | p.2 ≤ |p.1| ∧ |p.1| ≥ 1} := by
  sorry

end intersection_of_E_l1380_138071


namespace factory_sampling_is_systematic_l1380_138039

/-- Represents a sampling method -/
inductive SamplingMethod
| Simple
| Stratified
| Systematic

/-- Represents a sampling scenario -/
structure SamplingScenario where
  totalItems : Nat
  sampleSize : Nat
  method : SamplingMethod

/-- Determines if a sampling scenario is suitable for systematic sampling -/
def isSuitableForSystematic (scenario : SamplingScenario) : Prop :=
  scenario.method = SamplingMethod.Systematic ∧
  scenario.totalItems ≥ scenario.sampleSize ∧
  scenario.totalItems % scenario.sampleSize = 0

/-- The given sampling scenario -/
def factorySampling : SamplingScenario :=
  { totalItems := 2000
    sampleSize := 200
    method := SamplingMethod.Systematic }

/-- Theorem stating that the factory sampling scenario is suitable for systematic sampling -/
theorem factory_sampling_is_systematic :
  isSuitableForSystematic factorySampling :=
by
  sorry


end factory_sampling_is_systematic_l1380_138039


namespace discriminant_always_positive_roots_as_triangle_legs_m_values_l1380_138062

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := x^2 - (2+3*m)*x + 2*m^2 + 5*m - 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2+3*m)^2 - 4*(2*m^2 + 5*m - 4)

-- Theorem 1: The discriminant is always positive for any real m
theorem discriminant_always_positive (m : ℝ) : discriminant m > 0 := by
  sorry

-- Define the condition for the roots being legs of a right-angled triangle
def roots_are_triangle_legs (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 
    quadratic_eq m x₁ = 0 ∧ 
    quadratic_eq m x₂ = 0 ∧ 
    x₁^2 + x₂^2 = (2 * Real.sqrt 7)^2

-- Theorem 2: When the roots are legs of a right-angled triangle with hypotenuse 2√7, m is either -2 or 8/5
theorem roots_as_triangle_legs_m_values : 
  ∀ m : ℝ, roots_are_triangle_legs m → (m = -2 ∨ m = 8/5) := by
  sorry

end discriminant_always_positive_roots_as_triangle_legs_m_values_l1380_138062


namespace stickers_per_pack_l1380_138041

/-- Proves that the number of stickers in each pack is 30 --/
theorem stickers_per_pack (
  num_packs : ℕ)
  (cost_per_sticker : ℚ)
  (total_cost : ℚ)
  (h1 : num_packs = 4)
  (h2 : cost_per_sticker = 1/10)
  (h3 : total_cost = 12) :
  (total_cost / cost_per_sticker) / num_packs = 30 := by
  sorry

#check stickers_per_pack

end stickers_per_pack_l1380_138041


namespace remainder_invariance_l1380_138098

theorem remainder_invariance (n : ℤ) : (n + 22) % 9 = 2 → (n + 31) % 9 = 2 := by
  sorry

end remainder_invariance_l1380_138098


namespace geometric_sequences_theorem_l1380_138096

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences where
  a : ℕ → ℝ
  b : ℕ → ℝ
  a_pos : a 1 > 0
  a_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  b_geom : ∀ n : ℕ, b (n + 1) = b n * (b 2 / b 1)
  diff_1 : b 1 - a 1 = 1
  diff_2 : b 2 - a 2 = 2
  diff_3 : b 3 - a 3 = 3

/-- The unique value of a_1 in the geometric sequences -/
def unique_a (gs : GeometricSequences) : ℝ := gs.a 1

/-- The statement to be proved -/
theorem geometric_sequences_theorem (gs : GeometricSequences) :
  unique_a gs = 1/3 ∧
  ¬∃ (a b : ℕ → ℝ) (q₁ q₂ : ℝ),
    (∀ n, a (n + 1) = a n * q₁) ∧
    (∀ n, b (n + 1) = b n * q₂) ∧
    ∃ (d : ℝ), d ≠ 0 ∧
    ∀ n : ℕ, n ≤ 3 →
      (b (n + 1) - a (n + 1)) - (b n - a n) = d :=
sorry

end geometric_sequences_theorem_l1380_138096


namespace inscribed_circle_exists_l1380_138029

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define the area of a polygon
def area (p : ConvexPolygon) : ℝ := sorry

-- Define the perimeter of a polygon
def perimeter (p : ConvexPolygon) : ℝ := sorry

-- Define a point inside a polygon
def PointInside (p : ConvexPolygon) : Type := sorry

-- Define the distance from a point to a side of the polygon
def distanceToSide (point : PointInside p) (side : sorry) : ℝ := sorry

-- Theorem statement
theorem inscribed_circle_exists (p : ConvexPolygon) (h : area p > 0) :
  ∃ (center : PointInside p), ∀ (side : sorry),
    distanceToSide center side ≥ (area p) / (perimeter p) := by
  sorry

end inscribed_circle_exists_l1380_138029


namespace smallest_m_is_26_l1380_138069

def S : Finset Nat := Finset.range 100

-- Define the property we want to prove
def has_divisor (A : Finset Nat) : Prop :=
  ∃ x ∈ A, x ∣ (A.prod (fun y => if y ≠ x then y else 1))

theorem smallest_m_is_26 : 
  (∀ A : Finset Nat, A ⊆ S → A.card = 26 → has_divisor A) ∧ 
  (∀ m < 26, ∃ A : Finset Nat, A ⊆ S ∧ A.card = m ∧ ¬has_divisor A) :=
sorry

end smallest_m_is_26_l1380_138069


namespace arithmetic_sequence_fourth_term_l1380_138046

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5. -/
theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_sum : a 3 + a 5 = 10)  -- Sum of third and fifth terms is 10
  : a 4 = 5 := by
  sorry

end arithmetic_sequence_fourth_term_l1380_138046


namespace quadratic_roots_result_l1380_138076

theorem quadratic_roots_result (k p : ℕ) (hk : k > 0) 
  (h_roots : ∃ (x₁ x₂ : ℕ), x₁ > 0 ∧ x₂ > 0 ∧ 
    (k - 1) * x₁^2 - p * x₁ + k = 0 ∧
    (k - 1) * x₂^2 - p * x₂ + k = 0 ∧
    x₁ ≠ x₂) :
  k^(k*p) * (p^p + k^k) + (p + k) = 1989 := by
  sorry

end quadratic_roots_result_l1380_138076


namespace rachel_lost_lives_l1380_138082

/- Define the initial number of lives -/
def initial_lives : ℕ := 10

/- Define the number of lives gained -/
def lives_gained : ℕ := 26

/- Define the final number of lives -/
def final_lives : ℕ := 32

/- Theorem: Rachel lost 4 lives in the hard part -/
theorem rachel_lost_lives :
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 4 := by
  sorry

end rachel_lost_lives_l1380_138082


namespace two_digit_sum_square_property_l1380_138078

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def satisfiesCondition (A : ℕ) : Prop :=
  (sumOfDigits A)^2 = sumOfDigits (A^2)

def isTwoDigit (A : ℕ) : Prop :=
  10 ≤ A ∧ A ≤ 99

theorem two_digit_sum_square_property :
  ∀ A : ℕ, isTwoDigit A →
    (satisfiesCondition A ↔ 
      A = 10 ∨ A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31) :=
by sorry

end two_digit_sum_square_property_l1380_138078


namespace painting_scheme_combinations_l1380_138067

def number_of_color_choices : ℕ := 10
def colors_to_choose : ℕ := 2
def number_of_texture_choices : ℕ := 3
def textures_to_choose : ℕ := 1

theorem painting_scheme_combinations :
  (number_of_color_choices.choose colors_to_choose) * (number_of_texture_choices.choose textures_to_choose) = 135 := by
  sorry

end painting_scheme_combinations_l1380_138067


namespace number_puzzle_l1380_138007

theorem number_puzzle : ∃ x : ℚ, (x / 4 + 15 = 4 * x - 15) ∧ (x = 8) := by
  sorry

end number_puzzle_l1380_138007


namespace dime_position_l1380_138012

/-- Represents the two possible coin values -/
inductive CoinValue : Type
  | nickel : CoinValue
  | dime : CoinValue

/-- Represents the two possible pocket locations -/
inductive Pocket : Type
  | left : Pocket
  | right : Pocket

/-- Returns the value of a coin in cents -/
def coinValue (c : CoinValue) : Nat :=
  match c with
  | CoinValue.nickel => 5
  | CoinValue.dime => 10

/-- Represents the arrangement of coins in pockets -/
structure CoinArrangement :=
  (leftCoin : CoinValue)
  (rightCoin : CoinValue)

/-- Calculates the sum based on the given formula -/
def calculateSum (arr : CoinArrangement) : Nat :=
  3 * (coinValue arr.rightCoin) + 2 * (coinValue arr.leftCoin)

/-- The main theorem to prove -/
theorem dime_position (arr : CoinArrangement) :
  Even (calculateSum arr) ↔ arr.rightCoin = CoinValue.dime :=
sorry

end dime_position_l1380_138012


namespace boys_without_pencils_l1380_138058

theorem boys_without_pencils (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)
  (h1 : total_students = 30)
  (h2 : total_boys = 18)
  (h3 : students_with_pencils = 25)
  (h4 : girls_with_pencils = 15) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by sorry

end boys_without_pencils_l1380_138058
