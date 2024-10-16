import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3610_361039

theorem equation_solution : 
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 ∧ x = -30 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3610_361039


namespace NUMINAMATH_CALUDE_jenny_toy_spending_l3610_361009

/-- Proves that Jenny spent $200 on toys for the cat in the first year. -/
theorem jenny_toy_spending (adoption_fee vet_cost monthly_food_cost total_months jenny_total_spent : ℕ) 
  (h1 : adoption_fee = 50)
  (h2 : vet_cost = 500)
  (h3 : monthly_food_cost = 25)
  (h4 : total_months = 12)
  (h5 : jenny_total_spent = 625) : 
  jenny_total_spent - (adoption_fee + vet_cost + monthly_food_cost * total_months) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_jenny_toy_spending_l3610_361009


namespace NUMINAMATH_CALUDE_dunkers_lineup_count_l3610_361071

/-- The number of players in the team -/
def team_size : ℕ := 15

/-- The number of players who refuse to play together -/
def excluded_players : ℕ := 3

/-- The size of a starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to select a lineup from the team, excluding the combinations
    where the excluded players play together -/
def valid_lineups : ℕ := 2277

theorem dunkers_lineup_count :
  (excluded_players * (Nat.choose (team_size - excluded_players) (lineup_size - 1))) +
  (Nat.choose (team_size - excluded_players) lineup_size) = valid_lineups :=
sorry

end NUMINAMATH_CALUDE_dunkers_lineup_count_l3610_361071


namespace NUMINAMATH_CALUDE_comic_book_frames_l3610_361063

/-- The number of frames in Julian's comic book -/
def total_frames : ℕ := 143

/-- The number of frames per page if Julian puts them equally on 13 pages -/
def frames_per_page : ℕ := 11

/-- The number of pages if Julian puts 11 frames on each page -/
def number_of_pages : ℕ := 13

/-- Theorem stating that the total number of frames is correct -/
theorem comic_book_frames : 
  total_frames = frames_per_page * number_of_pages :=
by sorry

end NUMINAMATH_CALUDE_comic_book_frames_l3610_361063


namespace NUMINAMATH_CALUDE_car_trading_profit_l3610_361050

theorem car_trading_profit (P : ℝ) (h : P > 0) : 
  let discount_rate : ℝ := 0.3
  let increase_rate : ℝ := 0.7
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - P
  profit / P = 0.19 := by sorry

end NUMINAMATH_CALUDE_car_trading_profit_l3610_361050


namespace NUMINAMATH_CALUDE_M_eq_real_l3610_361041

/-- The set of complex numbers Z satisfying (Z-1)^2 = |Z-1|^2 -/
def M : Set ℂ := {Z | (Z - 1)^2 = Complex.abs (Z - 1)^2}

/-- Theorem stating that M is equal to the set of real numbers -/
theorem M_eq_real : M = {Z : ℂ | Z.im = 0} := by sorry

end NUMINAMATH_CALUDE_M_eq_real_l3610_361041


namespace NUMINAMATH_CALUDE_unique_x_intercept_l3610_361032

/-- The parabola equation: x = -3y^2 + 2y + 3 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

/-- X-intercept occurs when y = 0 -/
def x_intercept : ℝ := parabola 0

/-- Theorem: The parabola has exactly one x-intercept -/
theorem unique_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_unique_x_intercept_l3610_361032


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l3610_361064

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 5 * (1 / y) → x + y = (6 * Real.sqrt 60) / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l3610_361064


namespace NUMINAMATH_CALUDE_sufficient_necessary_condition_l3610_361027

theorem sufficient_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x > 0 → x + 1/x > a) ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_sufficient_necessary_condition_l3610_361027


namespace NUMINAMATH_CALUDE_wasted_meat_price_l3610_361013

def minimum_wage : ℝ := 8
def fruit_veg_price : ℝ := 4
def bread_price : ℝ := 1.5
def janitorial_wage : ℝ := 10
def fruit_veg_weight : ℝ := 15
def bread_weight : ℝ := 60
def meat_weight : ℝ := 20
def overtime_hours : ℝ := 10
def james_work_hours : ℝ := 50

theorem wasted_meat_price (meat_price : ℝ) : meat_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_wasted_meat_price_l3610_361013


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3610_361020

/-- The interest rate problem --/
theorem interest_rate_problem (total_investment : ℝ) (total_interest : ℝ) 
  (amount_at_r : ℝ) (rate_known : ℝ) :
  total_investment = 6000 →
  total_interest = 624 →
  amount_at_r = 1800 →
  rate_known = 0.11 →
  ∃ (r : ℝ), 
    amount_at_r * r + (total_investment - amount_at_r) * rate_known = total_interest ∧
    r = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3610_361020


namespace NUMINAMATH_CALUDE_mod_fourteen_power_ninety_six_minus_eight_l3610_361019

theorem mod_fourteen_power_ninety_six_minus_eight :
  (5^96 - 8) % 14 = 7 := by
sorry

end NUMINAMATH_CALUDE_mod_fourteen_power_ninety_six_minus_eight_l3610_361019


namespace NUMINAMATH_CALUDE_courtyard_paving_l3610_361052

/-- Represents the dimensions of a rectangular area in centimeters -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℕ) : ℕ := m * 100

/-- The dimensions of the courtyard in meters -/
def courtyard_m : Dimensions := ⟨30, 16⟩

/-- The dimensions of the courtyard in centimeters -/
def courtyard_cm : Dimensions := ⟨meters_to_cm courtyard_m.length, meters_to_cm courtyard_m.width⟩

/-- The dimensions of a single brick in centimeters -/
def brick : Dimensions := ⟨20, 10⟩

/-- Calculates the number of bricks needed to cover an area -/
def bricks_needed (area_to_cover : ℕ) (brick_size : ℕ) : ℕ := area_to_cover / brick_size

theorem courtyard_paving :
  bricks_needed (area courtyard_cm) (area brick) = 24000 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l3610_361052


namespace NUMINAMATH_CALUDE_equation_equivalence_l3610_361094

theorem equation_equivalence :
  ∀ x : ℝ, (x - 1) / 0.2 - x / 0.5 = 1 ↔ 3 * x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3610_361094


namespace NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l3610_361095

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1 -/
theorem even_quadratic_implies_k_eq_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l3610_361095


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3610_361030

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3610_361030


namespace NUMINAMATH_CALUDE_diagonal_length_of_prism_l3610_361007

/-- 
Given a rectangular prism with dimensions x, y, and z,
if the projections of its diagonal on each plane (xy, xz, yz) have length √2,
then the length of the diagonal is √3.
-/
theorem diagonal_length_of_prism (x y z : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 + z^2 = 2)
  (h3 : y^2 + z^2 = 2) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_length_of_prism_l3610_361007


namespace NUMINAMATH_CALUDE_prob_red_then_black_standard_deck_l3610_361004

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- Definition of a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 26,
    black_cards := 26 }

/-- Probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.black_cards : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability for a standard deck -/
theorem prob_red_then_black_standard_deck :
  prob_red_then_black standard_deck = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_standard_deck_l3610_361004


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3610_361076

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 3 * (10 * x^2 + 10 * x + 15) - x * (10 * x - 55)
  ∃ x : ℝ, f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -29/8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l3610_361076


namespace NUMINAMATH_CALUDE_xyz_equals_one_l3610_361057

theorem xyz_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 4)
  (eq2 : y + 1/z = 1)
  (eq3 : z + 1/x = 7/3) :
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_one_l3610_361057


namespace NUMINAMATH_CALUDE_average_problem_l3610_361084

theorem average_problem (x : ℝ) : 
  (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750 → x = 1255 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3610_361084


namespace NUMINAMATH_CALUDE_clover_field_count_l3610_361079

theorem clover_field_count : ∀ (total : ℕ),
  (total : ℝ) * (20 / 100) * (25 / 100) = 25 →
  total = 500 := by
  sorry

end NUMINAMATH_CALUDE_clover_field_count_l3610_361079


namespace NUMINAMATH_CALUDE_sum_of_products_divisible_by_2011_l3610_361082

def P (A : Finset ℕ) : ℕ := A.prod id

theorem sum_of_products_divisible_by_2011 : 
  ∃ (S : Finset (Finset ℕ)), 
    S.card = Nat.choose 2010 99 ∧ 
    (∀ A ∈ S, A.card = 99 ∧ A ⊆ Finset.range 2011) ∧
    2011 ∣ S.sum P :=
by sorry

end NUMINAMATH_CALUDE_sum_of_products_divisible_by_2011_l3610_361082


namespace NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l3610_361060

theorem perfect_squares_between_50_and_200 : 
  (Finset.filter (fun n : ℕ => 50 < n^2 ∧ n^2 ≤ 200) (Finset.range 201)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l3610_361060


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3610_361034

-- Define the rectangle's properties
def rectangle_area : ℝ := 54.3
def rectangle_width : ℝ := 6

-- Theorem statement
theorem rectangle_length_proof :
  let length := rectangle_area / rectangle_width
  length = 9.05 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l3610_361034


namespace NUMINAMATH_CALUDE_initial_books_l3610_361002

theorem initial_books (x : ℚ) : 
  (1/2 * x + 3 = 23) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_l3610_361002


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3610_361086

theorem remainder_divisibility (n : ℤ) : n % 22 = 12 → (2 * n) % 22 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3610_361086


namespace NUMINAMATH_CALUDE_robins_initial_gum_pieces_robins_initial_gum_pieces_proof_l3610_361062

/-- Given that Robin now has 62 pieces of gum after receiving 44.0 pieces from her brother,
    prove that her initial number of gum pieces was 18. -/
theorem robins_initial_gum_pieces : ℝ → Prop :=
  fun initial_gum =>
    initial_gum + 44.0 = 62 →
    initial_gum = 18
    
/-- Proof of the theorem -/
theorem robins_initial_gum_pieces_proof : robins_initial_gum_pieces 18 := by
  sorry

end NUMINAMATH_CALUDE_robins_initial_gum_pieces_robins_initial_gum_pieces_proof_l3610_361062


namespace NUMINAMATH_CALUDE_direct_proportion_implies_m_eq_two_l3610_361005

/-- A function y of x is a direct proportion if it can be written as y = kx where k is a non-zero constant -/
def is_direct_proportion (y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, y x = k * x

/-- Given y = (m^2 + 2m)x^(m^2 - 3), if y is a direct proportion function of x, then m = 2 -/
theorem direct_proportion_implies_m_eq_two (m : ℝ) :
  is_direct_proportion (fun x => (m^2 + 2*m) * x^(m^2 - 3)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_implies_m_eq_two_l3610_361005


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l3610_361056

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -8 * x^2 + 4 * x - 3 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l3610_361056


namespace NUMINAMATH_CALUDE_purple_bows_count_l3610_361018

/-- Given a bag of bows with the following properties:
    - 1/4 of the bows are red
    - 1/3 of the bows are blue
    - 1/6 of the bows are purple
    - The remaining 60 bows are yellow
    This theorem proves that there are 40 purple bows. -/
theorem purple_bows_count (total : ℕ) (red blue purple yellow : ℕ) : 
  red + blue + purple + yellow = total →
  4 * red = total →
  3 * blue = total →
  6 * purple = total →
  yellow = 60 →
  purple = 40 := by
  sorry

#check purple_bows_count

end NUMINAMATH_CALUDE_purple_bows_count_l3610_361018


namespace NUMINAMATH_CALUDE_point_symmetry_l3610_361077

/-- Given a line l: 2x - y - 1 = 0 and two points A and A', 
    this theorem states that A' is symmetric to A about l. -/
theorem point_symmetry (x y : ℚ) : 
  let l := {(x, y) : ℚ × ℚ | 2 * x - y - 1 = 0}
  let A := (3, -2)
  let A' := (-13/5, 4/5)
  let midpoint := ((A'.1 + A.1) / 2, (A'.2 + A.2) / 2)
  (2 * midpoint.1 - midpoint.2 - 1 = 0) ∧ 
  ((A'.2 - A.2) / (A'.1 - A.1) * 2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_symmetry_l3610_361077


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3610_361049

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 20th term of the arithmetic sequence 8, 5, 2, ... -/
theorem twentieth_term_of_sequence : arithmeticSequence 8 (-3) 20 = -49 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3610_361049


namespace NUMINAMATH_CALUDE_sum_of_roots_l3610_361096

theorem sum_of_roots (a b : ℝ) 
  (ha : a^4 - 16*a^3 + 40*a^2 - 50*a + 25 = 0)
  (hb : b^4 - 24*b^3 + 216*b^2 - 720*b + 625 = 0) :
  a + b = 7 ∨ a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3610_361096


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3610_361017

theorem inequality_equivalence :
  ∀ a : ℝ, a > 0 →
  ((∀ t₁ t₂ t₃ t₄ : ℝ, t₁ > 0 → t₂ > 0 → t₃ > 0 → t₄ > 0 → 
    t₁ * t₂ * t₃ * t₄ = a^4 →
    (1 / Real.sqrt (1 + t₁)) + (1 / Real.sqrt (1 + t₂)) + 
    (1 / Real.sqrt (1 + t₃)) + (1 / Real.sqrt (1 + t₄)) ≤ 
    4 / Real.sqrt (1 + a))
  ↔ 
  (0 < a ∧ a ≤ 7/9)) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3610_361017


namespace NUMINAMATH_CALUDE_smallest_angle_through_point_l3610_361022

theorem smallest_angle_through_point (α : Real) : 
  (∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * k) ∧ 
  (∀ β : Real, β > 0 → 
    (Real.sin β = Real.sin (2 * Real.pi / 3) ∧ 
     Real.cos β = Real.cos (2 * Real.pi / 3)) → 
    α ≤ β) ↔ 
  (Real.sin α = Real.sin (2 * Real.pi / 3) ∧ 
   Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
   α > 0 ∧ 
   α < 2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_through_point_l3610_361022


namespace NUMINAMATH_CALUDE_triangle_BCD_properties_l3610_361031

/-- Triangle BCD with given properties -/
structure TriangleBCD where
  BC : ℝ
  CD : ℝ
  M : ℝ × ℝ  -- Point M on BD
  angle_BCM : ℝ
  angle_MCD : ℝ
  h_BC : BC = 3
  h_CD : CD = 5
  h_angle_BCM : angle_BCM = π / 4  -- 45°
  h_angle_MCD : angle_MCD = π / 3  -- 60°

/-- Theorem about the properties of Triangle BCD -/
theorem triangle_BCD_properties (t : TriangleBCD) :
  -- 1. Ratio of BM to MD
  (∃ k : ℝ, k > 0 ∧ t.M.1 / t.M.2 = Real.sqrt 6 / 5 * k) ∧
  -- 2. Length of BM
  t.M.1 = (Real.sqrt 3 * Real.sqrt (68 + 15 * Real.sqrt 2 * (Real.sqrt 3 - 1))) / (Real.sqrt 6 + 5) ∧
  -- 3. Length of MD
  t.M.2 = (5 * Real.sqrt (68 + 15 * Real.sqrt 2 * (Real.sqrt 3 - 1))) / (2 * Real.sqrt 3 + 5 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_BCD_properties_l3610_361031


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3610_361045

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (49 - b) + c / (81 - c) = 8) :
  6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 66 / 36 + 77 / 49 + 99 / 81 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3610_361045


namespace NUMINAMATH_CALUDE_cos_neg_570_deg_l3610_361026

theorem cos_neg_570_deg : Real.cos ((-570 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_570_deg_l3610_361026


namespace NUMINAMATH_CALUDE_distinct_residues_count_l3610_361058

theorem distinct_residues_count (n m : ℕ) (a b : ℕ → ℝ) :
  (∀ j ∈ Finset.range n, ∀ k ∈ Finset.range m, a j + b k ≠ 1) →
  (∀ j ∈ Finset.range (n-1), a j < a (j+1)) →
  (∀ k ∈ Finset.range (m-1), b k < b (k+1)) →
  a 0 = 0 →
  b 0 = 0 →
  (∀ j ∈ Finset.range n, 0 < a j ∧ a j < 1) →
  (∀ k ∈ Finset.range m, 0 < b k ∧ b k < 1) →
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => (a p.1 + b p.2) % 1) (Finset.product (Finset.range n) (Finset.range m))) ≥ m + n - 1 :=
by sorry


end NUMINAMATH_CALUDE_distinct_residues_count_l3610_361058


namespace NUMINAMATH_CALUDE_sock_pair_count_l3610_361042

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: There are 93 ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 5 brown, 4 blue, and 2 red socks -/
theorem sock_pair_count :
  differentColorPairs 5 5 4 2 = 93 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l3610_361042


namespace NUMINAMATH_CALUDE_total_students_is_240_l3610_361014

/-- The number of students from Know It All High School -/
def know_it_all_students : ℕ := 50

/-- The number of students from Karen High School -/
def karen_students : ℕ := (3 * know_it_all_students) / 5

/-- The combined number of students from Know It All High School and Karen High School -/
def combined_students : ℕ := know_it_all_students + karen_students

/-- The number of students from Novel Corona High School -/
def novel_corona_students : ℕ := 2 * combined_students

/-- The total number of students at the competition -/
def total_students : ℕ := combined_students + novel_corona_students

/-- Theorem stating that the total number of students at the competition is 240 -/
theorem total_students_is_240 : total_students = 240 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_240_l3610_361014


namespace NUMINAMATH_CALUDE_roots_quadratic_expression_zero_l3610_361033

theorem roots_quadratic_expression_zero (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^2 + α*(β^2 - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_expression_zero_l3610_361033


namespace NUMINAMATH_CALUDE_probability_all_even_simplified_l3610_361088

def total_slips : ℕ := 49
def even_slips : ℕ := 9
def draws : ℕ := 8

def probability_all_even : ℚ :=
  (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2) / (49 * 48 * 47 * 46 * 45 * 44 * 43 * 42)

theorem probability_all_even_simplified (h : probability_all_even = 5 / (6 * 47 * 23 * 45 * 11 * 43 * 7)) :
  probability_all_even = 5 / (6 * 47 * 23 * 45 * 11 * 43 * 7) := by
  sorry

end NUMINAMATH_CALUDE_probability_all_even_simplified_l3610_361088


namespace NUMINAMATH_CALUDE_circle_m_range_chord_length_m_neg_two_m_value_circle_through_origin_l3610_361000

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem 1: Range of m
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) ↔ (m < 1 ∨ m > 4) :=
sorry

-- Theorem 2: Chord length when m = -2
theorem chord_length_m_neg_two :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ (-2) ∧ circle_equation x₂ y₂ (-2) ∧
    line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 26 :=
sorry

-- Theorem 3: Value of m when circle with MN as diameter passes through origin
theorem m_value_circle_through_origin :
  ∃ m x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ m ∧ circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧ line_equation x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    m = 2/29 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_chord_length_m_neg_two_m_value_circle_through_origin_l3610_361000


namespace NUMINAMATH_CALUDE_hotel_price_difference_l3610_361069

-- Define the charges for single rooms at hotels P, R, and G
def single_room_P (r g : ℝ) : ℝ := 0.45 * r
def single_room_P' (r g : ℝ) : ℝ := 0.90 * g

-- Define the charges for double rooms at hotels P, R, and G
def double_room_P (r g : ℝ) : ℝ := 0.70 * r
def double_room_P' (r g : ℝ) : ℝ := 0.80 * g

-- Define the charges for suites at hotels P, R, and G
def suite_P (r g : ℝ) : ℝ := 0.60 * r
def suite_P' (r g : ℝ) : ℝ := 0.85 * g

theorem hotel_price_difference (r_single g_single r_double g_double : ℝ) :
  single_room_P r_single g_single = single_room_P' r_single g_single ∧
  double_room_P r_double g_double = double_room_P' r_double g_double →
  (r_single / g_single - 1) * 100 - (r_double / g_double - 1) * 100 = 85.71 :=
by sorry

end NUMINAMATH_CALUDE_hotel_price_difference_l3610_361069


namespace NUMINAMATH_CALUDE_beverly_bottle_caps_l3610_361085

/-- The number of groups Beverly's bottle caps can be organized into -/
def num_groups : ℕ := 7

/-- The number of bottle caps in each group -/
def caps_per_group : ℕ := 5

/-- The total number of bottle caps in Beverly's collection -/
def total_caps : ℕ := num_groups * caps_per_group

/-- Theorem stating that the total number of bottle caps is 35 -/
theorem beverly_bottle_caps : total_caps = 35 := by
  sorry

end NUMINAMATH_CALUDE_beverly_bottle_caps_l3610_361085


namespace NUMINAMATH_CALUDE_system_solution_l3610_361037

theorem system_solution : 
  ∀ x y : ℝ, 
    (3 * x + Real.sqrt (3 * x - y) + y = 6 ∧ 
     9 * x^2 + 3 * x - y - y^2 = 36) ↔ 
    ((x = 2 ∧ y = -3) ∨ (x = 6 ∧ y = -18)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3610_361037


namespace NUMINAMATH_CALUDE_distinct_permutations_with_repetition_l3610_361015

theorem distinct_permutations_with_repetition : 
  let total_elements : ℕ := 5
  let repeated_elements : ℕ := 3
  let factorial (n : ℕ) := Nat.factorial n
  factorial total_elements / (factorial repeated_elements * factorial 1 * factorial 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_with_repetition_l3610_361015


namespace NUMINAMATH_CALUDE_largest_non_composite_sum_l3610_361043

def is_composite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

theorem largest_non_composite_sum : 
  (∀ n : ℕ, n > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) ∧
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b) :=
sorry

end NUMINAMATH_CALUDE_largest_non_composite_sum_l3610_361043


namespace NUMINAMATH_CALUDE_johns_wife_notebooks_l3610_361035

/-- Proves the number of notebooks John's wife bought for each child --/
theorem johns_wife_notebooks (num_children : ℕ) (johns_notebooks_per_child : ℕ) (total_notebooks : ℕ) :
  num_children = 3 →
  johns_notebooks_per_child = 2 →
  total_notebooks = 21 →
  (total_notebooks - num_children * johns_notebooks_per_child) / num_children = 5 := by
sorry

end NUMINAMATH_CALUDE_johns_wife_notebooks_l3610_361035


namespace NUMINAMATH_CALUDE_run_time_around_square_field_l3610_361061

/-- Calculates the time taken for a boy to run around a square field -/
theorem run_time_around_square_field (side_length : ℝ) (speed_kmh : ℝ) : 
  side_length = 60 → speed_kmh = 9 → 
  (4 * side_length) / (speed_kmh * 1000 / 3600) = 96 := by
  sorry

#check run_time_around_square_field

end NUMINAMATH_CALUDE_run_time_around_square_field_l3610_361061


namespace NUMINAMATH_CALUDE_second_number_proof_l3610_361023

theorem second_number_proof (first second third : ℝ) : 
  first = 6 → 
  third = 22 → 
  (first + second + third) / 3 = 13 → 
  second = 11 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l3610_361023


namespace NUMINAMATH_CALUDE_francis_family_violins_l3610_361090

/-- The number of ukuleles in Francis' family --/
def num_ukuleles : ℕ := 2

/-- The number of guitars in Francis' family --/
def num_guitars : ℕ := 4

/-- The number of strings on each ukulele --/
def strings_per_ukulele : ℕ := 4

/-- The number of strings on each guitar --/
def strings_per_guitar : ℕ := 6

/-- The number of strings on each violin --/
def strings_per_violin : ℕ := 4

/-- The total number of strings among all instruments --/
def total_strings : ℕ := 40

/-- The number of violins in Francis' family --/
def num_violins : ℕ := 2

theorem francis_family_violins :
  num_violins * strings_per_violin = 
    total_strings - (num_ukuleles * strings_per_ukulele + num_guitars * strings_per_guitar) :=
by sorry

end NUMINAMATH_CALUDE_francis_family_violins_l3610_361090


namespace NUMINAMATH_CALUDE_existence_of_xyz_l3610_361024

theorem existence_of_xyz (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^(n-1) + y^n = z^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xyz_l3610_361024


namespace NUMINAMATH_CALUDE_car_average_mpg_l3610_361006

def initial_odometer : ℕ := 57300
def final_odometer : ℕ := 58300
def initial_gas : ℕ := 8
def second_gas : ℕ := 14
def final_gas : ℕ := 22

def total_distance : ℕ := final_odometer - initial_odometer
def total_gas : ℕ := initial_gas + second_gas + final_gas

def average_mpg : ℚ := total_distance / total_gas

theorem car_average_mpg :
  (round (average_mpg * 10) / 10 : ℚ) = 227/10 := by sorry

end NUMINAMATH_CALUDE_car_average_mpg_l3610_361006


namespace NUMINAMATH_CALUDE_vector_magnitude_l3610_361053

theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (2, -1)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 8^2) →
  (b.1^2 + b.2^2 = 7^2) :=
by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3610_361053


namespace NUMINAMATH_CALUDE_min_stamps_for_60_cents_l3610_361067

theorem min_stamps_for_60_cents : ∃ (s t : ℕ), 
  5 * s + 6 * t = 60 ∧ 
  s + t = 11 ∧
  ∀ (s' t' : ℕ), 5 * s' + 6 * t' = 60 → s + t ≤ s' + t' := by
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_60_cents_l3610_361067


namespace NUMINAMATH_CALUDE_order_of_abc_l3610_361081

theorem order_of_abc (a b c : ℝ) (ha : a = 2^(1/10)) (hb : b = (1/2)^(4/5)) (hc : c = (1/2)^(1/2)) :
  a > c ∧ c > b :=
sorry

end NUMINAMATH_CALUDE_order_of_abc_l3610_361081


namespace NUMINAMATH_CALUDE_class_age_problem_l3610_361048

theorem class_age_problem (total_students : ℕ) (total_avg_age : ℝ) 
  (group_a_students : ℕ) (group_a_avg_age : ℝ)
  (group_b_students : ℕ) (group_b_avg_age : ℝ) :
  total_students = 50 →
  total_avg_age = 24 →
  group_a_students = 15 →
  group_a_avg_age = 20 →
  group_b_students = 25 →
  group_b_avg_age = 25 →
  let group_c_students := total_students - (group_a_students + group_b_students)
  let total_age := total_students * total_avg_age
  let group_a_total_age := group_a_students * group_a_avg_age
  let group_b_total_age := group_b_students * group_b_avg_age
  let group_c_total_age := total_age - (group_a_total_age + group_b_total_age)
  let group_c_avg_age := group_c_total_age / group_c_students
  group_c_avg_age = 27.5 := by
    sorry

end NUMINAMATH_CALUDE_class_age_problem_l3610_361048


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l3610_361047

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l3610_361047


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3610_361029

/-- A pyramid with a regular hexagonal base and isosceles triangular lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_base_length : ℝ
  lateral_face_height : ℝ

/-- A cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  edge_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.edge_length ^ 3

/-- Conditions for the specific pyramid and inscribed cube -/
def specific_pyramid_and_cube : Prop :=
  ∃ (p : HexagonalPyramid) (c : InscribedCube),
    p.base_side_length = 1 ∧
    p.lateral_face_base_length = 1 ∧
    c.pyramid = p ∧
    c.edge_length = 1

/-- Theorem stating that the volume of the inscribed cube is 1 -/
theorem inscribed_cube_volume :
  specific_pyramid_and_cube →
  ∃ (c : InscribedCube), cube_volume c = 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3610_361029


namespace NUMINAMATH_CALUDE_product_and_difference_imply_sum_l3610_361099

theorem product_and_difference_imply_sum (x y : ℕ+) : 
  x * y = 24 → x - y = 5 → x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_and_difference_imply_sum_l3610_361099


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3610_361008

theorem complex_absolute_value (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3610_361008


namespace NUMINAMATH_CALUDE_opposite_values_theorem_l3610_361038

theorem opposite_values_theorem (a b : ℝ) 
  (h : |a - 2| + (b + 1)^2 = 0) : 
  b^a = 1 ∧ a^3 + b^15 = 7 := by sorry

end NUMINAMATH_CALUDE_opposite_values_theorem_l3610_361038


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3610_361066

/-- A rhombus with diagonals of 6 and 8 units has a perimeter of 20 units. -/
theorem rhombus_perimeter (d₁ d₂ : ℝ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) :
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  4 * side = 20 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3610_361066


namespace NUMINAMATH_CALUDE_min_photos_theorem_l3610_361021

theorem min_photos_theorem (n_girls n_boys : ℕ) (h_girls : n_girls = 4) (h_boys : n_boys = 8) :
  ∃ (min_photos : ℕ), min_photos = n_girls * n_boys + 1 ∧
  (∀ (num_photos : ℕ), num_photos ≥ min_photos →
    (∃ (photo : Fin num_photos → Fin (n_girls + n_boys) × Fin (n_girls + n_boys)),
      (∃ (i : Fin num_photos), (photo i).1 ≥ n_girls ∧ (photo i).2 ≥ n_girls) ∨
      (∃ (i : Fin num_photos), (photo i).1 < n_girls ∧ (photo i).2 < n_girls) ∨
      (∃ (i j : Fin num_photos), i ≠ j ∧ photo i = photo j))) :=
by sorry

end NUMINAMATH_CALUDE_min_photos_theorem_l3610_361021


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3610_361083

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 4 = 3) ∧
  (n % 5 = 4) ∧
  (n % 6 = 5) ∧
  (n % 7 = 6) ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → m ≥ n) ∧
  n = 419 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3610_361083


namespace NUMINAMATH_CALUDE_salary_change_l3610_361040

theorem salary_change (S : ℝ) : 
  S * (1 + 0.25) * (1 - 0.15) * (1 + 0.10) * (1 - 0.20) = S * 0.935 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l3610_361040


namespace NUMINAMATH_CALUDE_test_scores_sum_l3610_361092

/-- Given the scores of Bill, John, and Sue on a test, prove that their total sum is 160 points. -/
theorem test_scores_sum (bill john sue : ℕ) : 
  bill = john + 20 →   -- Bill scored 20 more points than John
  bill = sue / 2 →     -- Bill scored half as many points as Sue
  bill = 45 →          -- Bill received 45 points
  bill + john + sue = 160 := by
sorry

end NUMINAMATH_CALUDE_test_scores_sum_l3610_361092


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3610_361093

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 4√3, b = 12, and B = 60°, then A = 30° -/
theorem triangle_angle_measure (a b c A B C : ℝ) : 
  a = 4 * Real.sqrt 3 → 
  b = 12 → 
  B = 60 * π / 180 → 
  A = 30 * π / 180 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3610_361093


namespace NUMINAMATH_CALUDE_probability_same_result_is_seven_twentyfourths_l3610_361070

/-- Represents a 12-sided die with specific colored sides -/
structure TwelveSidedDie :=
  (purple : Nat)
  (green : Nat)
  (orange : Nat)
  (glittery : Nat)
  (total_sides : Nat)
  (is_valid : purple + green + orange + glittery = total_sides)

/-- Calculates the probability of two dice showing the same result -/
def probability_same_result (d : TwelveSidedDie) : Rat :=
  let p_purple := (d.purple : Rat) / d.total_sides
  let p_green := (d.green : Rat) / d.total_sides
  let p_orange := (d.orange : Rat) / d.total_sides
  let p_glittery := (d.glittery : Rat) / d.total_sides
  p_purple * p_purple + p_green * p_green + p_orange * p_orange + p_glittery * p_glittery

/-- Theorem: The probability of two 12-sided dice with specific colored sides showing the same result is 7/24 -/
theorem probability_same_result_is_seven_twentyfourths :
  let d : TwelveSidedDie := {
    purple := 3,
    green := 4,
    orange := 4,
    glittery := 1,
    total_sides := 12,
    is_valid := by simp
  }
  probability_same_result d = 7 / 24 := by
  sorry


end NUMINAMATH_CALUDE_probability_same_result_is_seven_twentyfourths_l3610_361070


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3610_361080

theorem polynomial_division_theorem (x : ℝ) :
  let dividend := 12 * x^3 + 20 * x^2 - 7 * x + 4
  let divisor := 3 * x + 4
  let quotient := 4 * x^2 + (4/3) * x - 37/9
  let remainder := 74/9
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3610_361080


namespace NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l3610_361098

/-- Given that one kilometer equals 1000 meters, prove that one cubic kilometer equals 1,000,000,000 cubic meters. -/
theorem cubic_kilometer_to_cubic_meters :
  (1 : ℝ) * (kilometer ^ 3) = 1000000000 * (meter ^ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l3610_361098


namespace NUMINAMATH_CALUDE_max_blanks_proof_l3610_361097

/-- The width of the plywood sheet -/
def plywood_width : ℕ := 22

/-- The height of the plywood sheet -/
def plywood_height : ℕ := 15

/-- The width of the rectangular blank -/
def blank_width : ℕ := 3

/-- The height of the rectangular blank -/
def blank_height : ℕ := 5

/-- The maximum number of rectangular blanks that can be cut from the plywood sheet -/
def max_blanks : ℕ := 22

theorem max_blanks_proof :
  (plywood_width * plywood_height) ≥ (max_blanks * blank_width * blank_height) ∧
  (plywood_width * plywood_height) < ((max_blanks + 1) * blank_width * blank_height) :=
by sorry

end NUMINAMATH_CALUDE_max_blanks_proof_l3610_361097


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_three_and_six_l3610_361010

/-- Represents a repeating decimal with a single repeating digit -/
def RepeatingDecimal (d : Nat) : ℚ := d / 9

/-- The sum of the repeating decimals 0.3333... and 0.6666... is equal to 1 -/
theorem sum_of_repeating_decimals_three_and_six :
  RepeatingDecimal 3 + RepeatingDecimal 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_three_and_six_l3610_361010


namespace NUMINAMATH_CALUDE_books_together_l3610_361065

/-- The number of books Sandy, Tim, and Benny have together after Benny lost some books. -/
def remaining_books (sandy_books tim_books lost_books : ℕ) : ℕ :=
  sandy_books + tim_books - lost_books

/-- Theorem stating the number of books Sandy, Tim, and Benny have together. -/
theorem books_together : remaining_books 10 33 24 = 19 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l3610_361065


namespace NUMINAMATH_CALUDE_first_repeat_l3610_361051

/-- The number of points on the circle -/
def n : ℕ := 2021

/-- The function that calculates the position of the nth marked point -/
def f (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The theorem stating that 66 is the smallest positive integer b such that
    there exists an a < b where f(a) ≡ f(b) (mod n) -/
theorem first_repeat : 
  ∀ b < 66, ¬∃ a < b, f a % n = f b % n ∧ 
  ∃ a < 66, f a % n = f 66 % n :=
sorry

end NUMINAMATH_CALUDE_first_repeat_l3610_361051


namespace NUMINAMATH_CALUDE_log_inequality_l3610_361091

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 6 / Real.log 3 →
  b = Real.log 10 / Real.log 5 →
  c = Real.log 14 / Real.log 7 →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3610_361091


namespace NUMINAMATH_CALUDE_f_simplification_and_increasing_intervals_l3610_361036

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem f_simplification_and_increasing_intervals :
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6)) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π - π / 3) (k * π + π / 6))) := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_and_increasing_intervals_l3610_361036


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l3610_361054

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (x y z w v : ℝ) : Prop :=
  y / x = z / y ∧ z / y = w / z ∧ w / z = v / w

theorem arithmetic_geometric_sequence_property :
  ∀ (a b m n : ℝ),
  is_arithmetic_sequence (-9) a (-1) →
  is_geometric_sequence (-9) m b n (-1) →
  a * b = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l3610_361054


namespace NUMINAMATH_CALUDE_next_two_pythagorean_triples_l3610_361059

/-- Given a sequence of Pythagorean triples, find the next two triples -/
theorem next_two_pythagorean_triples 
  (h1 : 3^2 + 4^2 = 5^2)
  (h2 : 5^2 + 12^2 = 13^2)
  (h3 : 7^2 + 24^2 = 25^2) :
  (9^2 + 40^2 = 41^2) ∧ (11^2 + 60^2 = 61^2) := by
  sorry

end NUMINAMATH_CALUDE_next_two_pythagorean_triples_l3610_361059


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_10am_l3610_361016

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 10, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 25, seconds := 45 }

theorem add_12345_seconds_to_10am :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_10am_l3610_361016


namespace NUMINAMATH_CALUDE_test_score_mode_l3610_361078

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Calculates the mode of a list of numbers -/
def mode (numbers : List ℕ) : ℕ := sorry

/-- The stem-and-leaf plot of the test scores -/
def testScores : List StemLeafEntry := [
  ⟨5, [0, 5, 5]⟩,
  ⟨6, [2, 2, 8]⟩,
  ⟨7, [0, 1, 5, 9]⟩,
  ⟨8, [1, 1, 3, 5, 5, 5]⟩,
  ⟨9, [2, 6, 6, 8]⟩,
  ⟨10, [0, 0]⟩
]

/-- Converts a stem-and-leaf plot to a list of scores -/
def stemLeafToScores (plot : List StemLeafEntry) : List ℕ := sorry

theorem test_score_mode :
  mode (stemLeafToScores testScores) = 85 := by
  sorry

end NUMINAMATH_CALUDE_test_score_mode_l3610_361078


namespace NUMINAMATH_CALUDE_blue_fish_ratio_l3610_361087

theorem blue_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) : 
  total_fish = 30 →
  blue_spotted_fish = 5 →
  (blue_spotted_fish : ℚ) / (total_fish : ℚ) = 1/6 →
  (2 * blue_spotted_fish : ℚ) / (total_fish : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_ratio_l3610_361087


namespace NUMINAMATH_CALUDE_hall_covering_cost_l3610_361028

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for the given hall dimensions and mat cost is Rs. 39,000 -/
theorem hall_covering_cost : 
  total_expenditure 20 15 5 60 = 39000 := by
  sorry

end NUMINAMATH_CALUDE_hall_covering_cost_l3610_361028


namespace NUMINAMATH_CALUDE_angle_C_is_105_degrees_l3610_361055

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem angle_C_is_105_degrees (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b = 3 * Real.sqrt 2)
  (h3 : t.B = π / 4) : -- 45° in radians
  t.C = 7 * π / 12 := -- 105° in radians
by sorry

end NUMINAMATH_CALUDE_angle_C_is_105_degrees_l3610_361055


namespace NUMINAMATH_CALUDE_units_digit_of_M_M_10_l3610_361044

-- Define the sequence M_n
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => M (n + 1) + M n

-- Function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M_10 : unitsDigit (M (M 10)) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M_10_l3610_361044


namespace NUMINAMATH_CALUDE_fifth_term_geometric_l3610_361003

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

/-- The fifth term of a geometric sequence with first term 5 and common ratio 3y is 405y^4 -/
theorem fifth_term_geometric (y : ℝ) :
  geometric_term 5 (3*y) 5 = 405 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_l3610_361003


namespace NUMINAMATH_CALUDE_basketball_league_games_l3610_361011

/-- The number of games played in a league --/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 5 games with every other team, 
    the total number of games played is 225. --/
theorem basketball_league_games : total_games 10 5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l3610_361011


namespace NUMINAMATH_CALUDE_raft_travel_time_l3610_361001

theorem raft_travel_time (downstream_time upstream_time : ℝ) 
  (h1 : downstream_time = 5)
  (h2 : upstream_time = 7) :
  let steamer_speed := (1 / downstream_time + 1 / upstream_time) / 2
  let current_speed := (1 / downstream_time - 1 / upstream_time) / 2
  1 / current_speed = 35 := by sorry

end NUMINAMATH_CALUDE_raft_travel_time_l3610_361001


namespace NUMINAMATH_CALUDE_S_divisible_by_4003_l3610_361075

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def S : ℕ := factorial 2001 + (List.range 2001).foldl (λ acc i => acc * (2002 + i)) 1

theorem S_divisible_by_4003 : S % 4003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_S_divisible_by_4003_l3610_361075


namespace NUMINAMATH_CALUDE_function_extrema_implies_interval_bounds_l3610_361089

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem function_extrema_implies_interval_bounds
  (a : ℝ)
  (h_nonneg : 0 ≤ a)
  (h_max : ∀ x ∈ Set.Icc 0 a, f x ≤ 3)
  (h_min : ∀ x ∈ Set.Icc 0 a, 2 ≤ f x)
  (h_max_achieved : ∃ x ∈ Set.Icc 0 a, f x = 3)
  (h_min_achieved : ∃ x ∈ Set.Icc 0 a, f x = 2) :
  a ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_implies_interval_bounds_l3610_361089


namespace NUMINAMATH_CALUDE_existence_of_positive_rationals_l3610_361074

theorem existence_of_positive_rationals (n : ℕ) (h : n ≥ 4) :
  ∃ (k : ℕ) (a : ℕ → ℚ),
    k ≥ 2 ∧
    (∀ i, i ∈ Finset.range k → a i > 0) ∧
    (Finset.sum (Finset.range k) a = n) ∧
    (Finset.prod (Finset.range k) a = n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_positive_rationals_l3610_361074


namespace NUMINAMATH_CALUDE_athlete_calorie_burn_l3610_361073

/-- Calculates the total calories burned by an athlete during exercise -/
theorem athlete_calorie_burn 
  (running_rate : ℕ) 
  (walking_rate : ℕ) 
  (total_time : ℕ) 
  (running_time : ℕ) 
  (h1 : running_rate = 10)
  (h2 : walking_rate = 4)
  (h3 : total_time = 60)
  (h4 : running_time = 35)
  (h5 : running_time ≤ total_time) :
  running_rate * running_time + walking_rate * (total_time - running_time) = 450 := by
  sorry

#check athlete_calorie_burn

end NUMINAMATH_CALUDE_athlete_calorie_burn_l3610_361073


namespace NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_sq_gt_4_l3610_361012

theorem x_gt_2_sufficient_not_necessary_for_x_sq_gt_4 :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧ 
  ¬(∀ x : ℝ, x^2 > 4 → x > 2) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_sq_gt_4_l3610_361012


namespace NUMINAMATH_CALUDE_investment_relationship_l3610_361025

def initial_AA : ℝ := 150
def initial_BB : ℝ := 100
def initial_CC : ℝ := 200

def year1_AA_change : ℝ := 0.10
def year1_BB_change : ℝ := -0.20
def year1_CC_change : ℝ := 0.05

def year2_AA_change : ℝ := -0.05
def year2_BB_change : ℝ := 0.15
def year2_CC_change : ℝ := -0.10

def final_AA : ℝ := initial_AA * (1 + year1_AA_change) * (1 + year2_AA_change)
def final_BB : ℝ := initial_BB * (1 + year1_BB_change) * (1 + year2_BB_change)
def final_CC : ℝ := initial_CC * (1 + year1_CC_change) * (1 + year2_CC_change)

theorem investment_relationship : final_BB < final_AA ∧ final_AA < final_CC := by
  sorry

end NUMINAMATH_CALUDE_investment_relationship_l3610_361025


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l3610_361072

theorem decimal_fraction_equality : (0.2^3) / (0.02^2) = 20 := by sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l3610_361072


namespace NUMINAMATH_CALUDE_sum_of_angles_in_figure_l3610_361068

-- Define the angles
def angle_A : ℝ := 34
def angle_B : ℝ := 80
def angle_C : ℝ := 24

-- Define x and y as real numbers (measures of angles)
variable (x y : ℝ)

-- Define the theorem
theorem sum_of_angles_in_figure (h1 : 0 ≤ x) (h2 : 0 ≤ y) : x + y = 132 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_angles_in_figure_l3610_361068


namespace NUMINAMATH_CALUDE_unique_stamp_arrangements_l3610_361046

/-- Represents the number of stamps of each denomination -/
def stamp_counts : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Represents the value of each stamp denomination -/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- A type to represent a stamp arrangement -/
structure StampArrangement where
  stamps : List Nat
  sum_to_ten : (stamps.sum = 10)

/-- Function to count unique arrangements -/
def count_unique_arrangements (stamps : List Nat) (values : List Nat) : Nat :=
  sorry

/-- The main theorem stating that there are 88 unique arrangements -/
theorem unique_stamp_arrangements :
  count_unique_arrangements stamp_counts stamp_values = 88 := by
  sorry

end NUMINAMATH_CALUDE_unique_stamp_arrangements_l3610_361046
