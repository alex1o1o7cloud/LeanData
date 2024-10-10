import Mathlib

namespace max_value_a_l1610_161008

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 2 * c)
  (h3 : c < 5 * d)
  (h4 : d < 50) :
  a ≤ 1460 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1460 ∧ 
    a' < 3 * b' ∧ 
    b' < 2 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
by
  sorry

end max_value_a_l1610_161008


namespace radical_axis_existence_l1610_161021

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | power p c1 = power p c2}

def intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, power p c1 = 0 ∧ power p c2 = 0

def line_of_centers (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (t * c1.center.1 + (1 - t) * c2.center.1, 
                           t * c1.center.2 + (1 - t) * c2.center.2)}

def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧
    ∀ q r : ℝ × ℝ, q ∈ l1 → r ∈ l2 → 
      (q.1 - p.1) * (r.1 - p.1) + (q.2 - p.2) * (r.2 - p.2) = 0

theorem radical_axis_existence (c1 c2 : Circle) :
  (intersect c1 c2 → 
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ power p1 c1 = 0 ∧ power p1 c2 = 0 ∧
                     power p2 c1 = 0 ∧ power p2 c2 = 0 ∧
                     radical_axis c1 c2 = {p : ℝ × ℝ | ∃ t : ℝ, p = (t * p1.1 + (1 - t) * p2.1, 
                                                                   t * p1.2 + (1 - t) * p2.2)}) ∧
  (¬intersect c1 c2 → 
    ∃ c3 : Circle, intersect c1 c3 ∧ intersect c2 c3 ∧
    ∃ p : ℝ × ℝ, power p c1 = power p c2 ∧ power p c2 = power p c3 ∧
    perpendicular (radical_axis c1 c2) (line_of_centers c1 c2) ∧
    p ∈ radical_axis c1 c2) :=
sorry

end radical_axis_existence_l1610_161021


namespace range_of_x_minus_cosy_l1610_161017

theorem range_of_x_minus_cosy (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) :
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ 1 + Real.sqrt 3 := by
  sorry

end range_of_x_minus_cosy_l1610_161017


namespace min_occupied_seats_for_150_proof_37_seats_for_150_l1610_161012

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats + 3) / 4

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

/-- Proves that 37 is the minimum number of occupied seats required
    for 150 total seats to ensure the next person sits next to someone. -/
theorem proof_37_seats_for_150 :
  ∀ n : ℕ, n < min_occupied_seats 150 →
    ∃ arrangement : Fin 150 → Bool,
      (∀ i : Fin 150, arrangement i = true → i.val < n) ∧
      ∃ j : Fin 150, (∀ k : Fin 150, k.val = j.val - 1 ∨ k.val = j.val + 1 → arrangement k = false) := by
  sorry

end min_occupied_seats_for_150_proof_37_seats_for_150_l1610_161012


namespace partial_fraction_decomposition_l1610_161085

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 →
    5 * x / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by sorry

end partial_fraction_decomposition_l1610_161085


namespace cubic_polynomial_roots_l1610_161096

theorem cubic_polynomial_roots (x₁ x₂ x₃ s t u : ℝ) 
  (h₁ : x₁ + x₂ + x₃ = s) 
  (h₂ : x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = t) 
  (h₃ : x₁ * x₂ * x₃ = u) : 
  (X : ℝ) → (X - x₁) * (X - x₂) * (X - x₃) = X^3 - s*X^2 + t*X - u := by
  sorry

end cubic_polynomial_roots_l1610_161096


namespace julies_savings_l1610_161014

-- Define the initial savings amount
variable (S : ℝ)

-- Define the interest rate
variable (r : ℝ)

-- Define the time period
def t : ℝ := 2

-- Define the simple interest earned
def simple_interest : ℝ := 120

-- Define the compound interest earned
def compound_interest : ℝ := 126

-- Theorem statement
theorem julies_savings :
  (simple_interest = (S / 2) * r * t) ∧
  (compound_interest = (S / 2) * ((1 + r)^t - 1)) →
  S = 1200 := by
sorry

end julies_savings_l1610_161014


namespace shortest_ribbon_length_l1610_161000

theorem shortest_ribbon_length (ribbon_length : ℕ) : 
  (ribbon_length % 2 = 0 ∧ ribbon_length % 5 = 0) → 
  ribbon_length ≥ 10 :=
by sorry

end shortest_ribbon_length_l1610_161000


namespace outfit_choices_l1610_161069

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each type of clothing -/
def num_colors : ℕ := 8

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items * num_items * num_items

/-- The number of outfit combinations where shirt and pants are the same color -/
def matching_combinations : ℕ := num_colors * num_items

/-- The number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - matching_combinations

theorem outfit_choices :
  valid_outfits = 448 :=
sorry

end outfit_choices_l1610_161069


namespace closest_to_N_div_M_l1610_161058

/-- Mersenne prime M -/
def M : ℕ := 2^127 - 1

/-- Mersenne prime N -/
def N : ℕ := 2^607 - 1

/-- Approximation of log_2 -/
def log2_approx : ℝ := 0.3010

/-- Theorem stating that 10^144 is closest to N/M among given options -/
theorem closest_to_N_div_M :
  let options : List ℝ := [10^140, 10^142, 10^144, 10^146]
  ∀ x ∈ options, |((N : ℝ) / M) - 10^144| ≤ |((N : ℝ) / M) - x| :=
sorry

end closest_to_N_div_M_l1610_161058


namespace chinese_character_sum_l1610_161043

theorem chinese_character_sum (a b c d : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  100 * a + 10 * b + c + 100 * c + 10 * b + d = 1000 * a + 100 * b + 10 * c + d →
  1000 * a + 100 * b + 10 * c + d = 18 := by
sorry

end chinese_character_sum_l1610_161043


namespace multiply_2a_3a_l1610_161080

theorem multiply_2a_3a (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by sorry

end multiply_2a_3a_l1610_161080


namespace john_savings_l1610_161077

/-- Calculates the yearly savings when splitting an apartment --/
def yearly_savings (old_rent : ℕ) (price_increase_percent : ℕ) (num_people : ℕ) : ℕ :=
  let new_rent := old_rent + old_rent * price_increase_percent / 100
  let individual_share := new_rent / num_people
  let monthly_savings := old_rent - individual_share
  monthly_savings * 12

/-- Theorem: John saves $7680 per year by splitting the new apartment --/
theorem john_savings : yearly_savings 1200 40 3 = 7680 := by
  sorry

end john_savings_l1610_161077


namespace rope_length_proof_l1610_161060

theorem rope_length_proof (L : ℝ) 
  (h1 : L - 42 > 0)  -- Ensures the first rope has positive remaining length
  (h2 : L - 12 > 0)  -- Ensures the second rope has positive remaining length
  (h3 : L - 12 = 4 * (L - 42)) : 2 * L = 104 := by
  sorry

end rope_length_proof_l1610_161060


namespace coefficient_x_cubed_is_twenty_l1610_161097

/-- The coefficient of x^3 in the expansion of (2x + 1/(4x))^5 -/
def coefficient_x_cubed : ℚ :=
  let a := 2  -- coefficient of x
  let b := 1 / 4  -- coefficient of 1/x
  let n := 5  -- exponent
  let k := (n - 3) / 2  -- power of x is n - 2k, so n - 2k = 3
  (n.choose k) * (a ^ (n - k)) * (b ^ k)

/-- Theorem stating that the coefficient of x^3 in (2x + 1/(4x))^5 is 20 -/
theorem coefficient_x_cubed_is_twenty : coefficient_x_cubed = 20 := by
  sorry

end coefficient_x_cubed_is_twenty_l1610_161097


namespace five_coins_all_heads_or_tails_prob_l1610_161067

/-- The probability of getting all heads or all tails when flipping n fair coins -/
def all_heads_or_tails_prob (n : ℕ) : ℚ :=
  2 / 2^n

/-- Theorem: The probability of getting all heads or all tails when flipping 5 fair coins is 1/16 -/
theorem five_coins_all_heads_or_tails_prob :
  all_heads_or_tails_prob 5 = 1/16 := by
  sorry

end five_coins_all_heads_or_tails_prob_l1610_161067


namespace M_equals_N_set_order_irrelevant_l1610_161033

-- Define the sets M and N
def M : Set ℕ := {3, 2}
def N : Set ℕ := {2, 3}

-- Theorem stating that M and N are equal
theorem M_equals_N : M = N := by
  sorry

-- Additional theorem to emphasize that order doesn't matter in sets
theorem set_order_irrelevant (A B : Set α) : 
  (∀ x, x ∈ A ↔ x ∈ B) → A = B := by
  sorry

end M_equals_N_set_order_irrelevant_l1610_161033


namespace symmetric_point_coordinates_l1610_161015

/-- Given point A(-3, 1, 4), prove that its symmetric point B with respect to the origin has coordinates (3, -1, -4). -/
theorem symmetric_point_coordinates :
  let A : ℝ × ℝ × ℝ := (-3, 1, 4)
  let B : ℝ × ℝ × ℝ := (3, -1, -4)
  (∀ (x y z : ℝ), (x, y, z) = A → (-x, -y, -z) = B) :=
by sorry

end symmetric_point_coordinates_l1610_161015


namespace range_of_a_l1610_161081

open Set Real

noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x^2 + 3)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - log x - a

def I : Set ℝ := Ioo 0 2
def J : Set ℝ := Icc 1 2

theorem range_of_a :
  {a : ℝ | ∀ x₁ ∈ I, ∃ x₂ ∈ J, f x₁ = g a x₂} = Icc (1/2) (4/3 - log 2) := by sorry

end range_of_a_l1610_161081


namespace product_mod_seventeen_is_zero_l1610_161084

theorem product_mod_seventeen_is_zero :
  (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end product_mod_seventeen_is_zero_l1610_161084


namespace hydropower_station_calculations_l1610_161023

-- Define constants
def generator_power : Real := 24.5 * 1000  -- in watts
def generator_voltage : Real := 350
def line_resistance : Real := 4
def power_loss_percentage : Real := 0.05
def user_voltage : Real := 220

-- Define the theorem
theorem hydropower_station_calculations :
  let line_current := Real.sqrt ((power_loss_percentage * generator_power) / line_resistance)
  let step_up_ratio := (generator_power / generator_voltage) / line_current
  let step_down_input_voltage := generator_voltage - line_current * line_resistance
  let step_down_ratio := step_down_input_voltage / user_voltage
  (line_current = 17.5) ∧
  (step_up_ratio = 4) ∧
  (step_down_ratio = 133 / 22) := by
  sorry

end hydropower_station_calculations_l1610_161023


namespace evaluate_expression_l1610_161094

theorem evaluate_expression : 10010 - 12 * 3 * 2 = 9938 := by
  sorry

end evaluate_expression_l1610_161094


namespace smallest_sum_of_sequence_l1610_161090

theorem smallest_sum_of_sequence (E F G H : ℕ+) : 
  (∃ d : ℤ, (F : ℤ) - (E : ℤ) = d ∧ (G : ℤ) - (F : ℤ) = d) →  -- arithmetic sequence condition
  (∃ r : ℚ, (G : ℚ) / (F : ℚ) = r ∧ (H : ℚ) / (G : ℚ) = r) →  -- geometric sequence condition
  (G : ℚ) / (F : ℚ) = 4 / 3 →                                -- given ratio
  E + F + G + H ≥ 43 :=
by sorry

end smallest_sum_of_sequence_l1610_161090


namespace six_containing_triangles_l1610_161065

/-- Represents a quadrilateral composed of small equilateral triangles -/
structure TriangleQuadrilateral where
  /-- The total number of small equilateral triangles in the quadrilateral -/
  total_triangles : ℕ
  /-- The number of small triangles per side of the largest equilateral triangle -/
  max_side_length : ℕ
  /-- Assertion that the total number of triangles is 18 -/
  h_total : total_triangles = 18

/-- Counts the number of equilateral triangles containing a marked triangle -/
def count_containing_triangles (q : TriangleQuadrilateral) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 6 equilateral triangles containing the marked triangle -/
theorem six_containing_triangles (q : TriangleQuadrilateral) :
  count_containing_triangles q = 6 :=
sorry

end six_containing_triangles_l1610_161065


namespace income_calculation_l1610_161044

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 3 = expenditure * 5 →
  income - expenditure = savings →
  savings = 4000 →
  income = 10000 := by
sorry

end income_calculation_l1610_161044


namespace perfect_square_trinomial_condition_l1610_161004

/-- A perfect square trinomial in the form x^2 + ax + 4 -/
def is_perfect_square_trinomial (a : ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, x^2 + a*x + 4 = (x + b)^2

/-- If x^2 + ax + 4 is a perfect square trinomial, then a = ±4 -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  is_perfect_square_trinomial a → a = 4 ∨ a = -4 := by
  sorry

end perfect_square_trinomial_condition_l1610_161004


namespace flower_problem_l1610_161076

theorem flower_problem (total : ℕ) (roses_fraction : ℚ) (carnations : ℕ) (tulips : ℕ) :
  total = 40 →
  roses_fraction = 2 / 5 →
  carnations = 14 →
  tulips = total - (roses_fraction * total + carnations) →
  tulips = 10 := by
sorry

end flower_problem_l1610_161076


namespace max_extra_time_matches_2016_teams_l1610_161038

/-- Represents a hockey tournament -/
structure HockeyTournament where
  num_teams : Nat
  regular_win_points : Nat
  regular_loss_points : Nat
  extra_time_win_points : Nat
  extra_time_loss_points : Nat

/-- The maximum number of matches that could have ended in extra time -/
def max_extra_time_matches (tournament : HockeyTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of extra time matches for the given tournament -/
theorem max_extra_time_matches_2016_teams 
  (tournament : HockeyTournament)
  (h1 : tournament.num_teams = 2016)
  (h2 : tournament.regular_win_points = 3)
  (h3 : tournament.regular_loss_points = 0)
  (h4 : tournament.extra_time_win_points = 2)
  (h5 : tournament.extra_time_loss_points = 1) :
  max_extra_time_matches tournament = 1512 :=
sorry

end max_extra_time_matches_2016_teams_l1610_161038


namespace percent_problem_l1610_161083

theorem percent_problem (x : ℝ) (h : 0.22 * x = 66) : x = 300 := by
  sorry

end percent_problem_l1610_161083


namespace percent_difference_l1610_161063

theorem percent_difference (y u w z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
  sorry

end percent_difference_l1610_161063


namespace arc_length_sector_l1610_161002

/-- The arc length of a circular sector with central angle 90° and radius 6 is 3π. -/
theorem arc_length_sector (θ : ℝ) (r : ℝ) (h1 : θ = 90) (h2 : r = 6) :
  (θ / 360) * (2 * Real.pi * r) = 3 * Real.pi :=
by sorry

end arc_length_sector_l1610_161002


namespace shared_focus_parabola_ellipse_l1610_161052

/-- Given a parabola and an ellipse that share a focus, prove the value of m in the ellipse equation -/
theorem shared_focus_parabola_ellipse (x y : ℝ) (m : ℝ) : 
  (x^2 = 2*y) →  -- Parabola equation
  (y^2/m + x^2/2 = 1) →  -- Ellipse equation
  (∃ f : ℝ × ℝ, f ∈ {p : ℝ × ℝ | p.1^2 = 2*p.2} ∩ {e : ℝ × ℝ | e.2^2/m + e.1^2/2 = 1}) →  -- Shared focus
  (m = 9/4) :=
by sorry

end shared_focus_parabola_ellipse_l1610_161052


namespace remainder_17_63_mod_7_l1610_161053

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_63_mod_7_l1610_161053


namespace worker_count_l1610_161047

theorem worker_count : ∃ (x : ℕ), 
  x > 0 ∧ 
  (7200 / x + 400) * (x - 3) = 7200 ∧ 
  x = 9 := by
sorry

end worker_count_l1610_161047


namespace comic_cost_theorem_l1610_161032

/-- Calculates the final cost of each comic book type after discount --/
def final_comic_cost (common_cards : ℕ) (uncommon_cards : ℕ) (rare_cards : ℕ)
  (common_value : ℚ) (uncommon_value : ℚ) (rare_value : ℚ)
  (standard_price : ℚ) (deluxe_price : ℚ) (limited_price : ℚ)
  (discount_threshold_low : ℚ) (discount_threshold_high : ℚ)
  (discount_low : ℚ) (discount_high : ℚ)
  (ratio_standard : ℕ) (ratio_deluxe : ℕ) (ratio_limited : ℕ) : ℚ :=
  sorry

theorem comic_cost_theorem (common_cards : ℕ) (uncommon_cards : ℕ) (rare_cards : ℕ)
  (common_value : ℚ) (uncommon_value : ℚ) (rare_value : ℚ)
  (standard_price : ℚ) (deluxe_price : ℚ) (limited_price : ℚ)
  (discount_threshold_low : ℚ) (discount_threshold_high : ℚ)
  (discount_low : ℚ) (discount_high : ℚ)
  (ratio_standard : ℕ) (ratio_deluxe : ℕ) (ratio_limited : ℕ) :
  common_cards = 1000 ∧ uncommon_cards = 750 ∧ rare_cards = 250 ∧
  common_value = 5/100 ∧ uncommon_value = 1/10 ∧ rare_value = 1/5 ∧
  standard_price = 4 ∧ deluxe_price = 8 ∧ limited_price = 12 ∧
  discount_threshold_low = 100 ∧ discount_threshold_high = 150 ∧
  discount_low = 5/100 ∧ discount_high = 1/10 ∧
  ratio_standard = 3 ∧ ratio_deluxe = 2 ∧ ratio_limited = 1 →
  final_comic_cost common_cards uncommon_cards rare_cards
    common_value uncommon_value rare_value
    standard_price deluxe_price limited_price
    discount_threshold_low discount_threshold_high
    discount_low discount_high
    ratio_standard ratio_deluxe ratio_limited = 6 :=
by sorry

end comic_cost_theorem_l1610_161032


namespace tetrahedron_inequality_l1610_161001

/-- Given a tetrahedron with product of opposite edges equal to 1,
    angles α, β, γ between opposite edges, and face circumradii R₁, R₂, R₃, R₄,
    prove that sin²α + sin²β + sin²γ ≥ 1/√(R₁R₂R₃R₄) -/
theorem tetrahedron_inequality
  (α β γ R₁ R₂ R₃ R₄ : ℝ)
  (h_positive : R₁ > 0 ∧ R₂ > 0 ∧ R₃ > 0 ∧ R₄ > 0)
  (h_product : ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l → 
    ∃ (a_ij a_kl : ℝ), a_ij * a_kl = 1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 ≥ 1 / Real.sqrt (R₁ * R₂ * R₃ * R₄) := by
  sorry

end tetrahedron_inequality_l1610_161001


namespace second_smallest_is_four_probability_l1610_161036

def set_size : ℕ := 15
def selection_size : ℕ := 8
def target_number : ℕ := 4

def favorable_outcomes : ℕ := 924
def total_outcomes : ℕ := 6435

theorem second_smallest_is_four_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 27 := by
  sorry

end second_smallest_is_four_probability_l1610_161036


namespace remainder_problem_l1610_161026

theorem remainder_problem (N : ℤ) (h : N % 296 = 75) : N % 37 = 1 := by
  sorry

end remainder_problem_l1610_161026


namespace sum_base6_1452_2354_l1610_161046

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- The main theorem: sum of 1452₆ and 2354₆ in base 6 is 4250₆ -/
theorem sum_base6_1452_2354 :
  decimalToBase6 (base6ToDecimal [1, 4, 5, 2] + base6ToDecimal [2, 3, 5, 4]) = [4, 2, 5, 0] := by
  sorry

end sum_base6_1452_2354_l1610_161046


namespace complex_modulus_range_l1610_161041

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = Complex.mk a 1) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end complex_modulus_range_l1610_161041


namespace julios_fishing_time_l1610_161070

/-- Julio's fishing problem -/
theorem julios_fishing_time (catch_rate : ℕ) (fish_lost : ℕ) (final_fish : ℕ) (h : ℕ) : 
  catch_rate = 7 → fish_lost = 15 → final_fish = 48 → 
  catch_rate * h - fish_lost = final_fish → h = 9 := by
sorry

end julios_fishing_time_l1610_161070


namespace solution_set_of_inequality_l1610_161010

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_monotone_increasing_on_positive (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem solution_set_of_inequality 
  (h_odd : is_odd f)
  (h_monotone : is_monotone_increasing_on_positive f)
  (h_f1 : f 1 = 0) :
  {x : ℝ | (f x - f (-x)) / x > 0} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end solution_set_of_inequality_l1610_161010


namespace complex_fraction_simplification_l1610_161061

theorem complex_fraction_simplification :
  let z₁ : ℂ := Complex.mk 5 7
  let z₂ : ℂ := Complex.mk 2 3
  z₁ / z₂ = Complex.mk (31 / 13) (-1 / 13) := by sorry

end complex_fraction_simplification_l1610_161061


namespace simplify_expression_l1610_161057

theorem simplify_expression : (2^5 + 7^3) * (2^3 - (-2)^2)^8 = 24576000 := by
  sorry

end simplify_expression_l1610_161057


namespace population_growth_factors_l1610_161031

/-- Represents a population of organisms -/
structure Population where
  density : ℝ
  genotypeFrequency : ℝ
  kValue : ℝ

/-- Factors affecting population growth -/
inductive GrowthFactor
  | BirthRate
  | DeathRate
  | CarryingCapacity

/-- Represents ideal conditions for population growth -/
def idealConditions : Prop := sorry

/-- Main factors affecting population growth under ideal conditions -/
def mainFactors : Set GrowthFactor := sorry

theorem population_growth_factors :
  idealConditions →
  mainFactors = {GrowthFactor.BirthRate, GrowthFactor.DeathRate} ∧
  GrowthFactor.CarryingCapacity ∉ mainFactors :=
sorry

end population_growth_factors_l1610_161031


namespace function_inequality_function_inequality_bounded_l1610_161087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin x - (1/2) * Real.cos (2*x) + a - 3/a + 1/2

theorem function_inequality (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
sorry

theorem function_inequality_bounded (a : ℝ) (h : a ≥ 2) :
  (∃ x, f a x ≤ 0) ↔ a ≥ 3 :=
sorry

end function_inequality_function_inequality_bounded_l1610_161087


namespace trevor_eggs_left_l1610_161054

/-- Given the number of eggs laid by each chicken and the number of eggs dropped,
    prove that the number of eggs Trevor has left is equal to the total number
    of eggs collected minus the number of eggs dropped. -/
theorem trevor_eggs_left (gertrude blanche nancy martha dropped : ℕ) :
  gertrude + blanche + nancy + martha - dropped =
  (gertrude + blanche + nancy + martha) - dropped :=
by sorry

end trevor_eggs_left_l1610_161054


namespace min_value_implications_l1610_161093

theorem min_value_implications (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
  (3 * a^2 + b^2 ≥ 3) ∧ (4 / (a + 1) + 1 / b ≥ 3) := by
  sorry

end min_value_implications_l1610_161093


namespace car_speed_problem_l1610_161050

/-- Proves that a car traveling for two hours with an average speed of 80 km/h
    and a second hour speed of 90 km/h must have a first hour speed of 70 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 90 →
  average_speed = 80 →
  average_speed = (first_hour_speed + second_hour_speed) / 2 →
  first_hour_speed = 70 := by
sorry


end car_speed_problem_l1610_161050


namespace short_trees_after_planting_l1610_161006

/-- The number of short trees in a park after planting new trees -/
def total_short_trees (initial_short_trees newly_planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + newly_planted_short_trees

/-- Theorem stating that the total number of short trees after planting
    is equal to the sum of initial short trees and newly planted short trees -/
theorem short_trees_after_planting
  (initial_short_trees : ℕ)
  (initial_tall_trees : ℕ)
  (newly_planted_short_trees : ℕ) :
  total_short_trees initial_short_trees newly_planted_short_trees =
  initial_short_trees + newly_planted_short_trees :=
by
  sorry

/-- Example calculation for the specific problem -/
def park_short_trees : ℕ :=
  total_short_trees 3 9

#eval park_short_trees

end short_trees_after_planting_l1610_161006


namespace garden_area_proof_l1610_161078

theorem garden_area_proof (x : ℝ) : 
  (x + 2) * (x + 3) = 182 → x^2 = 121 := by
  sorry

end garden_area_proof_l1610_161078


namespace triangle_perimeter_l1610_161045

/-- Given a triangle with sides in the ratio 1/2 : 1/3 : 1/4 and longest side 48 cm, its perimeter is 104 cm -/
theorem triangle_perimeter (a b c : ℝ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a / b = 3 / 2 ∧ a / c = 2 ∧ b / c = 4 / 3) (h3 : a = 48) : 
  a + b + c = 104 := by
sorry

end triangle_perimeter_l1610_161045


namespace fourth_rectangle_area_l1610_161086

theorem fourth_rectangle_area (a b c : ℕ) (h1 : a + b + c = 350) : 
  ∃ d : ℕ, d = 300 - (a + b + c) ∧ d = 50 := by
  sorry

#check fourth_rectangle_area

end fourth_rectangle_area_l1610_161086


namespace marks_quiz_goal_l1610_161022

theorem marks_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (as_earned : ℕ) (h1 : total_quizzes = 60) 
  (h2 : goal_percentage = 85 / 100) (h3 : completed_quizzes = 40) 
  (h4 : as_earned = 30) : 
  Nat.ceil (↑total_quizzes * goal_percentage) - as_earned ≥ total_quizzes - completed_quizzes := by
  sorry

end marks_quiz_goal_l1610_161022


namespace ellipse_equation_l1610_161048

/-- Given an ellipse centered at the origin with eccentricity e = 1/2, 
    and one of its foci coinciding with the focus of the parabola y^2 = -4x,
    prove that the equation of this ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (e : ℝ) (f : ℝ × ℝ) :
  e = (1 : ℝ) / 2 →
  f = (-1, 0) →
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 + y^2 / b^2 = 1 ∧
      e = (f.1^2 + f.2^2).sqrt / a) :=
by sorry

end ellipse_equation_l1610_161048


namespace smallest_a_for_sqrt_12a_integer_three_satisfies_condition_three_is_smallest_l1610_161051

theorem smallest_a_for_sqrt_12a_integer (a : ℕ) : 
  (∃ (n : ℕ), n > 0 ∧ n^2 = 12*a) → a ≥ 3 :=
sorry

theorem three_satisfies_condition : 
  ∃ (n : ℕ), n > 0 ∧ n^2 = 12*3 :=
sorry

theorem three_is_smallest : 
  ∀ (a : ℕ), a > 0 → (∃ (n : ℕ), n > 0 ∧ n^2 = 12*a) → a ≥ 3 :=
sorry

end smallest_a_for_sqrt_12a_integer_three_satisfies_condition_three_is_smallest_l1610_161051


namespace f_properties_l1610_161062

def f (x : ℝ) := x^3 - 3*x

theorem f_properties :
  (∀ y, (∃ x, x = 0 ∧ y = f x) → y = 0) ∧
  (∀ x, x < -1 → (∀ h > 0, f (x + h) > f x)) ∧
  (∀ x, x > 1 → (∀ h > 0, f (x + h) > f x)) ∧
  (∀ x, -1 < x ∧ x < 1 → (∀ h > 0, f (x + h) < f x)) ∧
  (f (-1) = 2) ∧
  (f 1 = -2) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x ≥ -2) :=
sorry

end f_properties_l1610_161062


namespace percentage_increase_l1610_161049

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 1500 → final = 1800 → (final - initial) / initial * 100 = 20 := by
  sorry

end percentage_increase_l1610_161049


namespace group_size_calculation_l1610_161040

theorem group_size_calculation (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 3 ∧ weight_difference = 30 → 
  (weight_difference / average_increase : ℝ) = 10 := by
  sorry

#check group_size_calculation

end group_size_calculation_l1610_161040


namespace quadratic_two_roots_implies_a_le_two_l1610_161011

/-- 
Given a quadratic equation x^2 - 4x + 2a = 0 with parameter a,
if the equation has two real roots, then a ≤ 2.
-/
theorem quadratic_two_roots_implies_a_le_two : 
  ∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x^2 - 4*x + 2*a = 0 ∧ y^2 - 4*y + 2*a = 0) → a ≤ 2 :=
by sorry

end quadratic_two_roots_implies_a_le_two_l1610_161011


namespace lamppost_combinations_lamppost_problem_l1610_161056

theorem lamppost_combinations : Nat → Nat → Nat
| n, k => Nat.choose n k

theorem lamppost_problem :
  let total_posts : Nat := 11
  let posts_to_turn_off : Nat := 3
  let available_positions : Nat := total_posts - 4  -- Subtracting 2 for each end and 2 for adjacent positions
  lamppost_combinations available_positions posts_to_turn_off = 35 := by
  sorry

end lamppost_combinations_lamppost_problem_l1610_161056


namespace trig_simplification_l1610_161018

theorem trig_simplification :
  (Real.cos (40 * π / 180)) / (Real.cos (25 * π / 180) * Real.sqrt (1 - Real.sin (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end trig_simplification_l1610_161018


namespace complex_fraction_sum_complex_product_imaginary_l1610_161020

-- Problem 1
theorem complex_fraction_sum : (1 / (1 - Complex.I)) + (1 / (2 + 3 * Complex.I)) = Complex.mk (17/26) (7/26) := by sorry

-- Problem 2
theorem complex_product_imaginary (z₁ z₂ : ℂ) :
  z₁ = Complex.mk 3 4 →
  Complex.abs z₂ = 5 →
  (Complex.re (z₁ * z₂) = 0 ∧ Complex.im (z₁ * z₂) ≠ 0) →
  z₂ = Complex.mk 4 3 ∨ z₂ = Complex.mk (-4) (-3) := by sorry

end complex_fraction_sum_complex_product_imaginary_l1610_161020


namespace point_b_coordinates_l1610_161095

/-- Given points A and C, and the condition that vector AB is -2 times vector BC,
    prove that the coordinates of point B are (-2, -1). -/
theorem point_b_coordinates (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  C = (0, 1) → 
  B - A = -2 * (C - B) →
  B = (-2, -1) := by
sorry

end point_b_coordinates_l1610_161095


namespace bucket_weight_l1610_161074

/-- Given a bucket with weight p when three-quarters full and weight q when one-third full,
    prove that its weight when full is (8p - 7q) / 5 -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let x := (5 * q - 4 * p) / 5  -- weight of empty bucket
  let y := (12 * (p - q)) / 5   -- weight of water when bucket is full
  let weight_three_quarters := x + 3/4 * y
  let weight_one_third := x + 1/3 * y
  have h1 : weight_three_quarters = p := by sorry
  have h2 : weight_one_third = q := by sorry
  (8 * p - 7 * q) / 5

#check bucket_weight

end bucket_weight_l1610_161074


namespace monic_quartic_polynomial_value_l1610_161059

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (p 0)

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  is_monic_quartic p →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 6 = 163 := by
sorry

end monic_quartic_polynomial_value_l1610_161059


namespace roots_vs_ellipse_l1610_161005

def has_two_positive_roots (m n : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - m*x₁ + n = 0 ∧ x₂^2 - m*x₂ + n = 0

def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem roots_vs_ellipse (m n : ℝ) :
  ¬(has_two_positive_roots m n → is_ellipse m n) ∧
  ¬(is_ellipse m n → has_two_positive_roots m n) :=
sorry

end roots_vs_ellipse_l1610_161005


namespace jasmine_carry_weight_l1610_161027

/-- The weight of a bag of chips in ounces -/
def chipBagWeight : ℕ := 20

/-- The weight of a tin of cookies in ounces -/
def cookieTinWeight : ℕ := 9

/-- The number of bags of chips Jasmine buys -/
def numChipBags : ℕ := 6

/-- The ratio of tins of cookies to bags of chips Jasmine buys -/
def cookieToChipRatio : ℕ := 4

/-- The number of ounces in a pound -/
def ouncesPerPound : ℕ := 16

/-- Theorem: Given the conditions, Jasmine has to carry 21 pounds -/
theorem jasmine_carry_weight :
  (numChipBags * chipBagWeight +
   numChipBags * cookieToChipRatio * cookieTinWeight) / ouncesPerPound = 21 := by
  sorry

end jasmine_carry_weight_l1610_161027


namespace savings_ratio_l1610_161025

def debt : ℕ := 40
def lulu_savings : ℕ := 6
def nora_savings : ℕ := 5 * lulu_savings
def remaining_per_person : ℕ := 2

theorem savings_ratio (tamara_savings : ℕ) 
  (h1 : nora_savings + lulu_savings + tamara_savings = debt + 3 * remaining_per_person) :
  nora_savings / tamara_savings = 3 := by
  sorry

end savings_ratio_l1610_161025


namespace circle_equation_and_tangent_lines_l1610_161016

def circle_C (a b : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + (y - b)^2 = 2}

theorem circle_equation_and_tangent_lines :
  ∀ (a b : ℝ),
    b = a + 1 →
    (5 - a)^2 + (4 - b)^2 = 2 →
    (3 - a)^2 + (6 - b)^2 = 2 →
    (∃ (x y : ℝ), circle_C a b (x, y)) →
    (circle_C 4 5 = circle_C a b) ∧
    (∀ (k : ℝ),
      (k = 1 ∨ k = 23/7) ↔
      (∃ (x : ℝ), x ≠ 1 ∧ circle_C 4 5 (x, k*(x-1)) ∧
        ∀ (y : ℝ), y ≠ k*(x-1) → ¬ circle_C 4 5 (x, y))) :=
by sorry

end circle_equation_and_tangent_lines_l1610_161016


namespace f_properties_imply_m_range_l1610_161079

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the set of valid m values
def valid_m : Set ℝ := {m | m ≤ -2 ∨ m ≥ 2 ∨ m = 0}

theorem f_properties_imply_m_range :
  (∀ x, x ∈ [-1, 1] → f (-x) = -f x) →  -- f is odd
  f 1 = 1 →  -- f(1) = 1
  (∀ a b, a ∈ [-1, 1] → b ∈ [-1, 1] → a + b ≠ 0 → (f a + f b) / (a + b) > 0) →  -- given inequality
  (∀ m, (∀ x a, x ∈ [-1, 1] → a ∈ [-1, 1] → f x ≤ m^2 - 2*a*m + 1) ↔ m ∈ valid_m) :=
by sorry

end f_properties_imply_m_range_l1610_161079


namespace inverse_not_in_M_log_in_M_iff_exp_plus_square_in_M_l1610_161068

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1}

-- Theorem 1
theorem inverse_not_in_M :
  (fun x => 1 / x) ∉ M := sorry

-- Theorem 2
theorem log_in_M_iff (a : ℝ) :
  (fun x => Real.log (a / (x^2 + 1))) ∈ M ↔ 
  3 - Real.sqrt 5 ≤ a ∧ a ≤ 3 + Real.sqrt 5 := sorry

-- Theorem 3
theorem exp_plus_square_in_M :
  (fun x => 2^x + x^2) ∈ M := sorry

end inverse_not_in_M_log_in_M_iff_exp_plus_square_in_M_l1610_161068


namespace increasing_sequence_condition_l1610_161009

theorem increasing_sequence_condition (a : ℕ → ℝ) (b : ℝ) :
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) →
  (∀ n : ℕ, n > 0 → a n = n^2 + b*n) →
  b > -3 :=
sorry

end increasing_sequence_condition_l1610_161009


namespace max_sum_fraction_min_sum_fraction_l1610_161099

def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The maximum value of A/B + C/P given six different digits from Digits -/
theorem max_sum_fraction (A B C P Q R : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
                B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
                C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
                P ≠ Q ∧ P ≠ R ∧
                Q ≠ R)
  (h_in_digits : A ∈ Digits ∧ B ∈ Digits ∧ C ∈ Digits ∧ 
                 P ∈ Digits ∧ Q ∈ Digits ∧ R ∈ Digits) :
  (A : ℚ) / B + (C : ℚ) / P ≤ 13 :=
sorry

/-- The minimum value of Q/R + P/C using the remaining digits -/
theorem min_sum_fraction (A B C P Q R : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
                B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
                C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
                P ≠ Q ∧ P ≠ R ∧
                Q ≠ R)
  (h_in_digits : A ∈ Digits ∧ B ∈ Digits ∧ C ∈ Digits ∧ 
                 P ∈ Digits ∧ Q ∈ Digits ∧ R ∈ Digits) :
  (Q : ℚ) / R + (P : ℚ) / C ≥ 23 / 21 :=
sorry

end max_sum_fraction_min_sum_fraction_l1610_161099


namespace consecutive_divisible_numbers_exist_l1610_161019

theorem consecutive_divisible_numbers_exist : ∃ (n : ℕ),
  (∀ (i : Fin 11), ∃ (k : ℕ), n + i.val = k * (2 * i.val + 1)) :=
by sorry

end consecutive_divisible_numbers_exist_l1610_161019


namespace toy_car_factory_ratio_l1610_161066

/-- The ratio of cars made today to cars made yesterday -/
def car_ratio (cars_yesterday cars_today : ℕ) : ℚ :=
  cars_today / cars_yesterday

theorem toy_car_factory_ratio : 
  let cars_yesterday : ℕ := 60
  let total_cars : ℕ := 180
  let cars_today : ℕ := total_cars - cars_yesterday
  car_ratio cars_yesterday cars_today = 2 := by
sorry

end toy_car_factory_ratio_l1610_161066


namespace marathon_day_three_miles_l1610_161024

/-- Calculates the miles run on the third day of a three-day running schedule -/
def milesOnDayThree (totalMiles : ℝ) (day1Percent : ℝ) (day2Percent : ℝ) : ℝ :=
  let day1Miles := totalMiles * day1Percent
  let remainingAfterDay1 := totalMiles - day1Miles
  let day2Miles := remainingAfterDay1 * day2Percent
  totalMiles - day1Miles - day2Miles

/-- Theorem stating that given the specific conditions, the miles run on day 3 is 28 -/
theorem marathon_day_three_miles :
  milesOnDayThree 70 0.2 0.5 = 28 := by
  sorry

#eval milesOnDayThree 70 0.2 0.5

end marathon_day_three_miles_l1610_161024


namespace horse_grazing_width_l1610_161035

/-- Represents a rectangular field with a horse tethered to one corner. -/
structure GrazingField where
  length : ℝ
  width : ℝ
  rope_length : ℝ
  grazing_area : ℝ

/-- Theorem stating the width of the field that the horse can graze. -/
theorem horse_grazing_width (field : GrazingField)
  (h_length : field.length = 45)
  (h_rope : field.rope_length = 22)
  (h_area : field.grazing_area = 380.132711084365)
  : field.width = 22 := by
  sorry

end horse_grazing_width_l1610_161035


namespace arithmetic_sequence_property_l1610_161039

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arithmetic_property : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sum_of_terms seq 9 = 54 → 2 + seq.a 4 + 9 = 307 / 27 := by
  sorry

end arithmetic_sequence_property_l1610_161039


namespace z_in_second_quadrant_l1610_161042

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition
axiom z_condition : Complex.I^3 * z = 2 + Complex.I

-- Theorem to prove
theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l1610_161042


namespace angle_measure_proof_l1610_161029

theorem angle_measure_proof (x : ℝ) (h1 : x + (3 * x + 10) = 90) : x = 20 := by
  sorry

end angle_measure_proof_l1610_161029


namespace tangent_line_intercept_l1610_161007

/-- Given a curve y = ax + ln x with a tangent line y = 2x + b at the point (1, a), prove that b = -1 -/
theorem tangent_line_intercept (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x => a * x + Real.log x) →  -- Curve definition
  (∃ (g : ℝ → ℝ), g = λ x => 2 * x + b) →           -- Tangent line definition
  (∃ (x₀ : ℝ), x₀ = 1 ∧ f x₀ = a) →                 -- Point of tangency
  (∀ x, (deriv f) x = a + 1 / x) →                  -- Derivative of f
  (deriv f) 1 = 2 →                                 -- Slope at x = 1 equals 2
  b = -1 := by
sorry

end tangent_line_intercept_l1610_161007


namespace algebraic_multiplication_l1610_161088

theorem algebraic_multiplication (x y : ℝ) : 
  6 * x * y^2 * (-1/2 * x^3 * y^3) = -3 * x^4 * y^5 := by
  sorry

end algebraic_multiplication_l1610_161088


namespace inequalities_check_l1610_161089

theorem inequalities_check :
  (∀ x : ℝ, x^2 + 3 > 2*x) ∧
  (∃ a b : ℝ, a^5 + b^5 < a^3*b^2 + a^2*b^3) ∧
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*(a - b - 1)) :=
by sorry

end inequalities_check_l1610_161089


namespace max_roses_for_1000_budget_l1610_161073

/-- Represents the price of roses for different quantities -/
structure RosePrices where
  individual : ℚ
  dozen : ℚ
  two_dozen : ℚ
  five_dozen : ℚ
  hundred : ℚ

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def maxRoses (prices : RosePrices) (budget : ℚ) : ℕ :=
  sorry

/-- The theorem stating that given the specific rose prices and a $1000 budget, 
    the maximum number of roses that can be purchased is 548 -/
theorem max_roses_for_1000_budget :
  let prices : RosePrices := {
    individual := 5.3,
    dozen := 36,
    two_dozen := 50,
    five_dozen := 110,
    hundred := 180
  }
  maxRoses prices 1000 = 548 := by
  sorry

end max_roses_for_1000_budget_l1610_161073


namespace isosceles_triangle_angles_l1610_161055

/-- An isosceles triangle with one angle of 40 degrees has two equal angles of 70 degrees each. -/
theorem isosceles_triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal (isosceles property)
  c = 40 →           -- The third angle is 40°
  a = 70 :=          -- Each of the two equal angles is 70°
sorry

end isosceles_triangle_angles_l1610_161055


namespace intersection_determines_a_l1610_161092

theorem intersection_determines_a (A B : Set ℝ) (a : ℝ) :
  A = {1, 3, a} →
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := by
sorry

end intersection_determines_a_l1610_161092


namespace triangle_problem_l1610_161064

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.C.cos = 2 * t.A.cos * (t.B - π/6).sin)
  (h2 : t.S = 2 * Real.sqrt 3)
  (h3 : t.b - t.c = 2) :
  t.A = π/3 ∧ t.a = 2 * Real.sqrt 3 := by
  sorry


end triangle_problem_l1610_161064


namespace square_area_from_perimeter_l1610_161098

/-- The area in square centimeters of a square with perimeter 28 dm is 4900 -/
theorem square_area_from_perimeter : 
  let perimeter : ℝ := 28
  let side_length : ℝ := perimeter / 4
  let area_dm : ℝ := side_length ^ 2
  let area_cm : ℝ := area_dm * 100
  area_cm = 4900 := by
sorry

end square_area_from_perimeter_l1610_161098


namespace y_coordinate_of_point_p_l1610_161075

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def foci_distance : ℝ := 6

-- Define the sum of distances from P to foci
def sum_distances_to_foci : ℝ := 10

-- Define the radius of the inscribed circle
def inscribed_circle_radius : ℝ := 1

-- Main theorem
theorem y_coordinate_of_point_p :
  ∀ x y : ℝ,
  is_on_ellipse x y →
  x ≥ 0 →
  y > 0 →
  y = 8/3 :=
by sorry

end y_coordinate_of_point_p_l1610_161075


namespace ap_terms_count_l1610_161037

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Odd n → 
  (n + 1) / 2 * (2 * a + ((n + 1) / 2 - 1) * d) = 30 →
  (n - 1) / 2 * (2 * (a + d) + ((n - 1) / 2 - 1) * d) = 36 →
  n / 2 * (2 * a + (n - 1) * d) = 66 →
  a + (n - 1) * d - a = 12 →
  n = 9 := by
sorry

end ap_terms_count_l1610_161037


namespace pet_food_cost_differences_l1610_161072

/-- Calculates the total cost including tax -/
def totalCostWithTax (quantity : Float) (price : Float) (taxRate : Float) : Float :=
  quantity * price * (1 + taxRate)

/-- Theorem: Sum of differences between pet food costs -/
theorem pet_food_cost_differences (dogQuantity catQuantity birdQuantity fishQuantity : Float)
  (dogPrice catPrice birdPrice fishPrice : Float) (taxRate : Float)
  (h1 : dogQuantity = 600.5)
  (h2 : catQuantity = 327.25)
  (h3 : birdQuantity = 415.75)
  (h4 : fishQuantity = 248.5)
  (h5 : dogPrice = 24.99)
  (h6 : catPrice = 19.49)
  (h7 : birdPrice = 15.99)
  (h8 : fishPrice = 13.89)
  (h9 : taxRate = 0.065) :
  let dogCost := totalCostWithTax dogQuantity dogPrice taxRate
  let catCost := totalCostWithTax catQuantity catPrice taxRate
  let birdCost := totalCostWithTax birdQuantity birdPrice taxRate
  let fishCost := totalCostWithTax fishQuantity fishPrice taxRate
  (dogCost - catCost) + (catCost - birdCost) + (birdCost - fishCost) = 12301.9002 :=
by sorry

end pet_food_cost_differences_l1610_161072


namespace equation_solution_l1610_161003

theorem equation_solution : ∃ x : ℝ, 3 * x + 6 = |(-23 + 9)| ∧ x = 8/3 := by
  sorry

end equation_solution_l1610_161003


namespace jake_has_seven_balls_l1610_161028

/-- The number of balls Audrey has -/
def audrey_balls : ℕ := 41

/-- The difference in the number of balls between Audrey and Jake -/
def difference : ℕ := 34

/-- The number of balls Jake has -/
def jake_balls : ℕ := audrey_balls - difference

/-- Theorem stating that Jake has 7 balls -/
theorem jake_has_seven_balls : jake_balls = 7 := by
  sorry

end jake_has_seven_balls_l1610_161028


namespace power_of_negative_power_l1610_161082

theorem power_of_negative_power (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end power_of_negative_power_l1610_161082


namespace greatest_number_with_special_remainder_l1610_161071

theorem greatest_number_with_special_remainder : ∃ n : ℕ, 
  (n % 91 = (n / 91) ^ 2) ∧ 
  (∀ m : ℕ, m > n → m % 91 ≠ (m / 91) ^ 2) ∧
  n = 900 := by
sorry

end greatest_number_with_special_remainder_l1610_161071


namespace gold_bar_distribution_l1610_161091

theorem gold_bar_distribution (initial_bars : ℕ) (lost_bars : ℕ) (friends : ℕ) 
  (h1 : initial_bars = 100)
  (h2 : lost_bars = 20)
  (h3 : friends = 4)
  (h4 : friends > 0) :
  (initial_bars - lost_bars) / friends = 20 := by
  sorry

end gold_bar_distribution_l1610_161091


namespace new_barbell_cost_l1610_161034

def old_barbell_cost : ℝ := 250
def price_increase_percentage : ℝ := 30

theorem new_barbell_cost : 
  old_barbell_cost * (1 + price_increase_percentage / 100) = 325 := by
  sorry

end new_barbell_cost_l1610_161034


namespace stamp_arrangement_count_l1610_161013

/-- Represents a stamp with its value in cents -/
structure Stamp where
  value : Nat
  deriving Repr

/-- Represents an arrangement of stamps -/
def Arrangement := List Stamp

/-- Checks if an arrangement is valid (sums to 15 cents) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  (arr.map (·.value)).sum = 15

/-- Checks if two arrangements are considered the same -/
def isSameArrangement (arr1 arr2 : Arrangement) : Bool :=
  sorry  -- Implementation details omitted

/-- Generates all possible stamp arrangements -/
def generateArrangements (stamps : List (Nat × Nat)) : List Arrangement :=
  sorry  -- Implementation details omitted

/-- Counts unique arrangements -/
def countUniqueArrangements (arrangements : List Arrangement) : Nat :=
  sorry  -- Implementation details omitted

/-- The main theorem to prove -/
theorem stamp_arrangement_count :
  let stamps := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
  let arrangements := generateArrangements stamps
  let validArrangements := arrangements.filter isValidArrangement
  countUniqueArrangements validArrangements = 48 := by
  sorry

end stamp_arrangement_count_l1610_161013


namespace jims_investment_l1610_161030

/-- 
Given an investment scenario with three investors and a total investment,
calculate the investment amount for one specific investor.
-/
theorem jims_investment
  (total_investment : ℕ) 
  (john_ratio : ℕ) 
  (james_ratio : ℕ) 
  (jim_ratio : ℕ) 
  (h1 : total_investment = 80000)
  (h2 : john_ratio = 4)
  (h3 : james_ratio = 7)
  (h4 : jim_ratio = 9) : 
  jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 := by
  sorry

end jims_investment_l1610_161030
