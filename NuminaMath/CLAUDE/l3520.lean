import Mathlib

namespace pigeonhole_principle_sports_choices_l3520_352013

/-- Given a set of 50 people, each making choices from three categories with 4, 3, and 2 options respectively,
    there must be at least 3 people who have made exactly the same choices for all three categories. -/
theorem pigeonhole_principle_sports_choices :
  ∀ (choices : Fin 50 → Fin 4 × Fin 3 × Fin 2),
  ∃ (c : Fin 4 × Fin 3 × Fin 2) (s₁ s₂ s₃ : Fin 50),
  s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧
  choices s₁ = c ∧ choices s₂ = c ∧ choices s₃ = c :=
by sorry

end pigeonhole_principle_sports_choices_l3520_352013


namespace james_beef_cost_l3520_352040

def beef_purchase (num_packs : ℕ) (weight_per_pack : ℝ) (price_per_pound : ℝ) : ℝ :=
  (num_packs : ℝ) * weight_per_pack * price_per_pound

theorem james_beef_cost :
  beef_purchase 5 4 5.50 = 110 := by
  sorry

end james_beef_cost_l3520_352040


namespace apartment_333_on_third_floor_l3520_352029

/-- Represents a building with apartments -/
structure Building where
  floors : ℕ
  entrances : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in the building -/
def total_apartments (b : Building) : ℕ :=
  b.floors * b.entrances * b.apartments_per_floor

/-- Calculates the floor number for a given apartment number -/
def apartment_floor (b : Building) (apartment_number : ℕ) : ℕ :=
  ((apartment_number - 1) / b.apartments_per_floor) % b.floors + 1

/-- The specific building described in the problem -/
def problem_building : Building :=
  { floors := 9
  , entrances := 10
  , apartments_per_floor := 4 }

theorem apartment_333_on_third_floor :
  apartment_floor problem_building 333 = 3 := by
  sorry

#eval apartment_floor problem_building 333

end apartment_333_on_third_floor_l3520_352029


namespace square_perimeter_division_l3520_352037

/-- Represents a division of a square's perimeter into two groups of segments -/
structure SquarePerimeterDivision where
  side_length : ℝ
  group1_count : ℕ
  group2_count : ℕ
  group1_segment_length : ℝ
  group2_segment_length : ℝ

/-- Checks if the given division is valid for the square's perimeter -/
def is_valid_division (d : SquarePerimeterDivision) : Prop :=
  d.group1_count * d.group1_segment_length + d.group2_count * d.group2_segment_length = 4 * d.side_length

/-- The specific division of a square with side length 20 cm into 3 and 4 segments -/
def specific_division : SquarePerimeterDivision :=
  { side_length := 20
  , group1_count := 3
  , group2_count := 4
  , group1_segment_length := 20
  , group2_segment_length := 5 }

theorem square_perimeter_division :
  is_valid_division specific_division ∧
  specific_division.group1_segment_length = 20 ∧
  specific_division.group2_segment_length = 5 := by
  sorry

#check square_perimeter_division

end square_perimeter_division_l3520_352037


namespace soccer_team_average_goals_l3520_352012

/-- The average number of goals scored by the soccer team per game -/
def team_average_goals (carter_goals shelby_goals judah_goals : ℝ) : ℝ :=
  carter_goals + shelby_goals + judah_goals

/-- Theorem stating the average total number of goals scored by the team per game -/
theorem soccer_team_average_goals :
  ∃ (carter_goals shelby_goals judah_goals : ℝ),
    carter_goals = 4 ∧
    shelby_goals = carter_goals / 2 ∧
    judah_goals = 2 * shelby_goals - 3 ∧
    team_average_goals carter_goals shelby_goals judah_goals = 7 := by
  sorry


end soccer_team_average_goals_l3520_352012


namespace equation_solution_l3520_352075

def equation (x : ℝ) : Prop :=
  (45 * x)^2 = (0.45 * 1200) * 80 / (12 + 4 * 3)

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ abs (x - 0.942808153803174) < 1e-10 := by
  sorry

end equation_solution_l3520_352075


namespace value_added_to_numbers_l3520_352060

theorem value_added_to_numbers (n : ℕ) (original_avg new_avg x : ℝ) 
  (h1 : n = 15)
  (h2 : original_avg = 40)
  (h3 : new_avg = 53)
  (h4 : n * new_avg = n * original_avg + n * x) :
  x = 13 := by
  sorry

end value_added_to_numbers_l3520_352060


namespace arithmetic_geometric_sequence_values_l3520_352008

theorem arithmetic_geometric_sequence_values :
  ∀ (a b c : ℝ),
  (∃ (d : ℝ), b = (a + c) / 2 ∧ c - b = b - a) →  -- arithmetic sequence condition
  (a + b + c = 12) →  -- sum condition
  (∃ (r : ℝ), (b + 2) ^ 2 = (a + 2) * (c + 5)) →  -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by sorry

end arithmetic_geometric_sequence_values_l3520_352008


namespace claire_earnings_l3520_352097

/-- Represents the total earnings from selling roses with discounts applied -/
def total_earnings (total_flowers : ℕ) (tulips : ℕ) (white_roses : ℕ) 
  (small_red_roses : ℕ) (medium_red_roses : ℕ) 
  (small_price : ℚ) (medium_price : ℚ) (large_price : ℚ) : ℚ :=
  let total_roses := total_flowers - tulips
  let red_roses := total_roses - white_roses
  let large_red_roses := red_roses - small_red_roses - medium_red_roses
  let small_sold := small_red_roses / 2
  let medium_sold := medium_red_roses / 2
  let large_sold := large_red_roses / 2
  let small_earnings := small_sold * small_price * (1 - 0.1)  -- 10% discount
  let medium_earnings := medium_sold * medium_price * (1 - 0.15)  -- 15% discount
  let large_earnings := large_sold * large_price * (1 - 0.15)  -- 15% discount
  small_earnings + medium_earnings + large_earnings

/-- Theorem stating that Claire's earnings are $92.13 -/
theorem claire_earnings : 
  total_earnings 400 120 80 40 60 0.75 1 1.25 = 92.13 := by
  sorry


end claire_earnings_l3520_352097


namespace elizabeth_pencil_purchase_l3520_352066

/-- The amount of additional cents Elizabeth needs to purchase a pencil -/
def additional_cents_needed (elizabeth_dollars : ℕ) (borrowed_cents : ℕ) (pencil_dollars : ℕ) : ℕ :=
  pencil_dollars * 100 - (elizabeth_dollars * 100 + borrowed_cents)

theorem elizabeth_pencil_purchase :
  additional_cents_needed 5 53 6 = 47 := by
  sorry

end elizabeth_pencil_purchase_l3520_352066


namespace bucket_weight_l3520_352077

/-- Given a bucket with the following properties:
    1. When three-fourths full, it weighs p kilograms.
    2. When one-third full, it weighs q kilograms.
    This theorem states that when the bucket is five-sixths full, 
    it weighs (6p - q) / 5 kilograms. -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let weight_three_fourths := p
  let weight_one_third := q
  let weight_five_sixths := (6 * p - q) / 5
  weight_five_sixths

#check bucket_weight

end bucket_weight_l3520_352077


namespace f_increasing_iff_a_ge_five_l3520_352048

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

/-- Theorem stating the condition for f(x) to be increasing on (-∞, 4) -/
theorem f_increasing_iff_a_ge_five (a : ℝ) :
  (∀ x y, x < y ∧ y < 4 → f a x < f a y) ↔ a ≥ 5 :=
sorry

end f_increasing_iff_a_ge_five_l3520_352048


namespace square_unbounded_l3520_352076

theorem square_unbounded : ∀ (M : ℝ), M > 0 → ∃ (N : ℝ), ∀ (x : ℝ), x > N → x^2 > M := by
  sorry

end square_unbounded_l3520_352076


namespace sqrt_inequality_solution_set_l3520_352018

theorem sqrt_inequality_solution_set (x : ℝ) :
  x + 3 ≥ 0 →
  (Real.sqrt (x + 3) > 3 - x ↔ x > 1) :=
by sorry

end sqrt_inequality_solution_set_l3520_352018


namespace ellipse_sum_l3520_352092

theorem ellipse_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →  -- Ellipse equation
  (h = 3 ∧ k = -5) →                                   -- Center at (3, -5)
  (a = 7 ∨ b = 7) →                                    -- Semi-major axis is 7
  (a = 2 ∨ b = 2) →                                    -- Semi-minor axis is 2
  (a > b) →                                            -- Ensure a is semi-major axis
  h + k + a + b = 7 :=
by sorry

end ellipse_sum_l3520_352092


namespace units_digit_problem_l3520_352046

theorem units_digit_problem : (7 * 13 * 1957 - 7^4) % 10 = 6 := by
  sorry

end units_digit_problem_l3520_352046


namespace saturday_ice_cream_amount_l3520_352064

/-- The amount of ice cream eaten on Saturday night, given the amount eaten on Friday and the total amount eaten over both nights. -/
def ice_cream_saturday (friday : ℝ) (total : ℝ) : ℝ :=
  total - friday

theorem saturday_ice_cream_amount :
  ice_cream_saturday 3.25 3.5 = 0.25 := by
  sorry

end saturday_ice_cream_amount_l3520_352064


namespace officer_selection_ways_l3520_352044

def group_members : Nat := 5
def officer_positions : Nat := 4

theorem officer_selection_ways : 
  (group_members.choose officer_positions) * (officer_positions.factorial) = 120 := by
  sorry

end officer_selection_ways_l3520_352044


namespace can_detect_drum_l3520_352083

-- Define the stone type
def Stone : Type := ℕ

-- Define the set of 100 stones
def S : Finset Stone := sorry

-- Define the weight function
def weight : Stone → ℕ := sorry

-- Define the property that all stones have different weights
axiom different_weights : ∀ s₁ s₂ : Stone, s₁ ≠ s₂ → weight s₁ ≠ weight s₂

-- Define a subset of 10 stones
def Subset : Type := Finset Stone

-- Define the property that a subset has exactly 10 stones
def has_ten_stones (subset : Subset) : Prop := subset.card = 10

-- Define the ordering function (by the brownie)
def order_stones (subset : Subset) : List Stone := sorry

-- Define the potential swapping function (by the drum)
def swap_stones (ordered_stones : List Stone) : List Stone := sorry

-- Define the observation function (what Andryusha sees)
def observe (subset : Subset) : List Stone := sorry

-- The main theorem
theorem can_detect_drum :
  ∃ (f : Subset → Bool),
    (∀ subset : Subset, has_ten_stones subset →
      f subset = true ↔ observe subset ≠ order_stones subset) :=
sorry

end can_detect_drum_l3520_352083


namespace short_bingo_first_column_count_l3520_352014

def short_bingo_first_column_possibilities : ℕ := 360360

theorem short_bingo_first_column_count :
  (Finset.range 15).card.choose 5 = short_bingo_first_column_possibilities :=
by sorry

end short_bingo_first_column_count_l3520_352014


namespace sum_mean_median_mode_l3520_352000

def numbers : List ℝ := [-3, -1, 0, 2, 2, 3, 3, 3, 4, 5]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem sum_mean_median_mode :
  mean numbers + median numbers + mode numbers = 7.3 := by sorry

end sum_mean_median_mode_l3520_352000


namespace pig_problem_l3520_352094

theorem pig_problem (x y : ℕ) : 
  (y - 100 = 100 * x) →  -- If each person contributes 100 coins, there's a surplus of 100
  (y = 90 * x) →         -- If each person contributes 90 coins, it's just enough
  (x = 10 ∧ y = 900) :=  -- Then the number of people is 10 and the price of the pig is 900
by sorry

end pig_problem_l3520_352094


namespace beach_trip_driving_time_l3520_352091

theorem beach_trip_driving_time :
  ∀ (x : ℝ),
  (2.5 * (2 * x) + 2 * x = 14) →
  x = 2 :=
by
  sorry

end beach_trip_driving_time_l3520_352091


namespace constant_pace_running_time_l3520_352016

/-- Represents the time taken to run a certain distance at a constant pace -/
structure RunningPace where
  distance : ℝ
  time : ℝ

/-- Theorem: If it takes 24 minutes to run 3 miles at a constant pace, 
    then it will take 16 minutes to run 2 miles at the same pace -/
theorem constant_pace_running_time 
  (park : RunningPace) 
  (library : RunningPace) 
  (h1 : park.distance = 3) 
  (h2 : park.time = 24) 
  (h3 : library.distance = 2) 
  (h4 : park.time / park.distance = library.time / library.distance) : 
  library.time = 16 := by
sorry

end constant_pace_running_time_l3520_352016


namespace plot_perimeter_l3520_352090

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ
  length_width_relation : length = width + 10
  cost_equation : fencing_cost = (2 * (length + width)) * fencing_rate

/-- The perimeter of the rectangular plot is 300 meters -/
theorem plot_perimeter (plot : RectangularPlot) 
  (h : plot.fencing_rate = 6.5 ∧ plot.fencing_cost = 1950) : 
  2 * (plot.length + plot.width) = 300 := by
  sorry

end plot_perimeter_l3520_352090


namespace power_3_2023_mod_5_l3520_352015

theorem power_3_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end power_3_2023_mod_5_l3520_352015


namespace sum_of_roots_quadratic_l3520_352099

theorem sum_of_roots_quadratic (b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - b*x + 20
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x * y = 20) →
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x + y = b) :=
by sorry

end sum_of_roots_quadratic_l3520_352099


namespace triangle_shape_l3520_352096

theorem triangle_shape (A B C : Real) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  ∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a ^ 2 + b ^ 2 - c ^ 2 < 0 := by
  sorry

end triangle_shape_l3520_352096


namespace second_number_is_13_l3520_352019

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ        -- Total number of items
  sampleSize : ℕ   -- Number of items to be sampled
  first : ℕ        -- First number drawn

/-- Calculates the nth number in a systematic sample -/
def nthNumber (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first + (s.total / s.sampleSize) * (n - 1)

/-- Theorem stating that the second number drawn is 13 -/
theorem second_number_is_13 (s : SystematicSample) 
  (h1 : s.total = 500) 
  (h2 : s.sampleSize = 50) 
  (h3 : s.first = 3) : 
  nthNumber s 2 = 13 := by
  sorry

#check second_number_is_13

end second_number_is_13_l3520_352019


namespace clown_count_l3520_352071

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 357

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 842

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 300534 := by
  sorry

end clown_count_l3520_352071


namespace even_not_div_four_not_sum_consec_odd_l3520_352056

theorem even_not_div_four_not_sum_consec_odd (n : ℤ) : 
  ¬(∃ k : ℤ, 2 * (n + 1) = 4 * k + 2) :=
sorry

end even_not_div_four_not_sum_consec_odd_l3520_352056


namespace min_tank_cost_l3520_352039

/-- Represents the cost function for a rectangular water tank. -/
def tank_cost (x y : ℝ) : ℝ :=
  120 * (x * y) + 100 * (2 * 3 * x + 2 * 3 * y)

/-- Theorem stating the minimum cost for the water tank construction. -/
theorem min_tank_cost :
  let volume : ℝ := 300
  let depth : ℝ := 3
  let bottom_cost : ℝ := 120
  let wall_cost : ℝ := 100
  ∀ x y : ℝ,
    x > 0 → y > 0 →
    x * y * depth = volume →
    tank_cost x y ≥ 24000 ∧
    (x = 10 ∧ y = 10 → tank_cost x y = 24000) :=
by sorry

end min_tank_cost_l3520_352039


namespace quadratic_equation_solution_l3520_352042

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = -3/2 ∧ 
  (∀ x : ℝ, 2*x*(x-5) = 3*(5-x) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end quadratic_equation_solution_l3520_352042


namespace cubic_equation_roots_l3520_352005

theorem cubic_equation_roots (m : ℝ) : 
  (m = 3 ∨ m = -2) → 
  ∃ (z₁ z₂ z₃ : ℝ), 
    z₁ = -1 ∧ z₂ = -3 ∧ z₃ = 4 ∧
    ∀ (z : ℝ), z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6) = 0 ↔ 
      (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end cubic_equation_roots_l3520_352005


namespace marias_paper_count_l3520_352098

theorem marias_paper_count : 
  ∀ (desk_sheets backpack_sheets : ℕ),
    desk_sheets = 50 →
    backpack_sheets = 41 →
    desk_sheets + backpack_sheets = 91 :=
by
  sorry

end marias_paper_count_l3520_352098


namespace problem_solution_l3520_352058

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (h₁ : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (h₂ : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12)
  (h₃ : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 :=
by sorry

end problem_solution_l3520_352058


namespace rancher_cows_count_l3520_352081

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →  -- The rancher raises 5 times as many cows as horses
  cows + horses = 168 →  -- The total number of animals is 168
  cows = 140 :=  -- Prove that the number of cows is 140
by sorry

end rancher_cows_count_l3520_352081


namespace cone_volume_and_surface_area_l3520_352036

/-- Given a cone with slant height 17 cm and height 15 cm, prove its volume and lateral surface area -/
theorem cone_volume_and_surface_area :
  let slant_height : ℝ := 17
  let height : ℝ := 15
  let radius : ℝ := Real.sqrt (slant_height ^ 2 - height ^ 2)
  let volume : ℝ := (1 / 3) * π * radius ^ 2 * height
  let lateral_surface_area : ℝ := π * radius * slant_height
  volume = 320 * π ∧ lateral_surface_area = 136 * π := by
  sorry


end cone_volume_and_surface_area_l3520_352036


namespace exists_monthly_increase_factor_l3520_352023

/-- The marathon distance in miles -/
def marathon_distance : ℝ := 26.3

/-- The initial running distance in miles -/
def initial_distance : ℝ := 3

/-- The number of months of training -/
def training_months : ℕ := 5

/-- Theorem stating the existence of a monthly increase factor -/
theorem exists_monthly_increase_factor :
  ∃ x : ℝ, x > 1 ∧ initial_distance * x^(training_months - 1) = marathon_distance :=
sorry

end exists_monthly_increase_factor_l3520_352023


namespace min_perimeter_isosceles_triangles_l3520_352026

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 4 : ℝ) * Real.sqrt (4 * t.leg^2 - t.base^2 : ℝ)

/-- Theorem stating the minimum common perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base * 8 = t2.base * 9 ∧
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    perimeter t1 = perimeter t2 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1.base * 8 = s2.base * 9 →
      s1 ≠ s2 →
      area s1 = area s2 →
      perimeter s1 = perimeter s2 →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 960 :=
  sorry

end min_perimeter_isosceles_triangles_l3520_352026


namespace necessary_but_not_sufficient_l3520_352052

-- Define the conditions
def condition_p (m : ℝ) : Prop := ∃ x : ℝ, |x - 1| + |x - 3| < m

def condition_q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (7 - 3*m)^x > (7 - 3*m)^y

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, condition_q m → condition_p m) ∧
  (∃ m : ℝ, condition_p m ∧ ¬condition_q m) :=
sorry

end necessary_but_not_sufficient_l3520_352052


namespace rationalize_denominator_l3520_352050

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l3520_352050


namespace bowling_team_size_l3520_352086

theorem bowling_team_size (original_average : ℝ) (new_average : ℝ) 
  (new_player1_weight : ℝ) (new_player2_weight : ℝ) 
  (h1 : original_average = 76) 
  (h2 : new_average = 78) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) : 
  ∃ n : ℕ, n > 0 ∧ 
  (n : ℝ) * original_average + new_player1_weight + new_player2_weight = 
  (n + 2 : ℝ) * new_average := by
  sorry

#check bowling_team_size

end bowling_team_size_l3520_352086


namespace line_points_l3520_352047

-- Define the points
def p1 : ℝ × ℝ := (8, 10)
def p2 : ℝ × ℝ := (2, -2)

-- Define the function to check if a point is on the line
def is_on_line (p : ℝ × ℝ) : Prop :=
  let m := (p1.2 - p2.2) / (p1.1 - p2.1)
  let b := p1.2 - m * p1.1
  p.2 = m * p.1 + b

-- Theorem statement
theorem line_points :
  is_on_line (5, 4) ∧
  is_on_line (1, -4) ∧
  ¬is_on_line (4, 1) ∧
  ¬is_on_line (3, -1) ∧
  ¬is_on_line (6, 7) :=
by sorry

end line_points_l3520_352047


namespace video_game_expense_is_correct_l3520_352024

def total_allowance : ℚ := 50

def book_fraction : ℚ := 1/2
def toy_fraction : ℚ := 1/4
def snack_fraction : ℚ := 1/10

def video_game_expense : ℚ := total_allowance - (book_fraction * total_allowance + toy_fraction * total_allowance + snack_fraction * total_allowance)

theorem video_game_expense_is_correct : video_game_expense = 7.5 := by
  sorry

end video_game_expense_is_correct_l3520_352024


namespace quadratic_equation_solution_l3520_352079

theorem quadratic_equation_solution :
  ∃! (x : ℚ), x > 0 ∧ 6 * x^2 + 9 * x - 24 = 0 ∧ x = 4/3 := by
  sorry

end quadratic_equation_solution_l3520_352079


namespace union_A_complement_B_when_m_neg_two_A_implies_B_iff_A_not_B_iff_A_subset_complement_B_iff_l3520_352006

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m - 1) * (x - m - 7) > 0}

-- Theorem 1
theorem union_A_complement_B_when_m_neg_two :
  A ∪ (Set.univ \ B (-2)) = {x : ℝ | -2 < x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem A_implies_B_iff :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∈ B m) ↔ m ∈ Set.Iic (-9) ∪ Set.Ici 1 := by sorry

-- Theorem 3
theorem A_not_B_iff :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ A → x ∉ B m) ↔ m ∈ Set.Icc (-5) (-3) := by sorry

-- Theorem 4
theorem A_subset_complement_B_iff :
  ∀ m : ℝ, A ⊆ (Set.univ \ B m) ↔ m ∈ Set.Ioo (-5) (-3) := by sorry

end union_A_complement_B_when_m_neg_two_A_implies_B_iff_A_not_B_iff_A_subset_complement_B_iff_l3520_352006


namespace equation_root_one_l3520_352062

theorem equation_root_one (k : ℝ) : 
  let a : ℝ := 13 / 2
  let b : ℝ := -4
  ∃ x : ℝ, x = 1 ∧ (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 :=
by sorry

end equation_root_one_l3520_352062


namespace hard_drives_sold_l3520_352065

/-- Represents the number of hard drives sold -/
def num_hard_drives : ℕ := 14

/-- Represents the total earnings from all items -/
def total_earnings : ℕ := 8960

/-- Theorem stating that the number of hard drives sold is 14 -/
theorem hard_drives_sold : 
  10 * 600 + 8 * 200 + 4 * 60 + num_hard_drives * 80 = total_earnings :=
by sorry

end hard_drives_sold_l3520_352065


namespace parabola_properties_l3520_352001

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c m : ℝ) :
  -- Conditions
  (∃ C : ℝ, C > 0 ∧ parabola a b c C = 0) →  -- Intersects positive y-axis
  (parabola a b c 1 = 2) →                   -- Vertex at (1, 2)
  (parabola a b c (-1) = m) →                -- Passes through (-1, m)
  (m < 0) →                                  -- m is negative
  -- Conclusions
  (2 * a + b = 0) ∧                          -- Conclusion ②
  (-2 < a ∧ a < -1/2) ∧                      -- Conclusion ③
  (∀ n : ℝ, (∀ x : ℝ, parabola a b c x ≠ n) → n > 2) ∧  -- Conclusion ④
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola a b c x₁ = 1 ∧ parabola a b c x₂ = 1 ∧ x₁ + x₂ = 2)  -- Conclusion ⑥
  := by sorry

end parabola_properties_l3520_352001


namespace complex_equality_problem_l3520_352041

theorem complex_equality_problem (x y : ℝ) 
  (h : (x + y : ℂ) + Complex.I = 3*x + (x - y)*Complex.I) : 
  x = -1 ∧ y = -2 := by
sorry

end complex_equality_problem_l3520_352041


namespace pure_imaginary_complex_number_l3520_352003

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a * (a - 1) + a * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l3520_352003


namespace leonards_age_l3520_352038

theorem leonards_age (nina jerome leonard : ℕ) 
  (h1 : leonard = nina - 4)
  (h2 : nina = jerome / 2)
  (h3 : nina + jerome + leonard = 36) : 
  leonard = 6 := by
sorry

end leonards_age_l3520_352038


namespace ratio_of_segments_l3520_352025

-- Define the right triangle
def right_triangle (a b c r s : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 ∧
  c^2 = a^2 + b^2 ∧
  c = r + s ∧
  a^2 = r * c ∧
  b^2 = s * c

-- Theorem statement
theorem ratio_of_segments (a b c r s : ℝ) :
  right_triangle a b c r s →
  a / b = 2 / 5 →
  r / s = 4 / 25 := by
  sorry

end ratio_of_segments_l3520_352025


namespace prob_roll_three_l3520_352034

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Nat)
  (fair : sides = 6)

/-- The probability of rolling a specific number on a fair die -/
def prob_roll (d : FairDie) (n : Nat) : ℚ :=
  1 / d.sides

/-- The sequence of previous rolls -/
def previous_rolls : List Nat := [6, 6, 6, 6, 6, 6]

/-- Theorem: The probability of rolling a 3 on a fair six-sided die is 1/6,
    regardless of previous rolls -/
theorem prob_roll_three (d : FairDie) (prev : List Nat) :
  prob_roll d 3 = 1 / 6 :=
sorry

end prob_roll_three_l3520_352034


namespace square_plus_reciprocal_square_l3520_352078

theorem square_plus_reciprocal_square (x : ℝ) (hx : x ≠ 0) 
  (h : x + 1/x = Real.sqrt 2019) : x^2 + 1/x^2 = 2017 := by
  sorry

end square_plus_reciprocal_square_l3520_352078


namespace certificate_recipients_l3520_352082

theorem certificate_recipients (total : ℕ) (difference : ℕ) (recipients : ℕ) : 
  total = 120 → 
  difference = 36 → 
  recipients = total / 2 + difference / 2 → 
  recipients = 78 := by
sorry

end certificate_recipients_l3520_352082


namespace pentagon_coverage_is_62_5_percent_l3520_352053

/-- Represents a tiling of the plane with large squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each row/column of a large square -/
  grid_size : ℕ
  /-- The number of smaller squares that are part of pentagons in each large square -/
  pentagon_squares : ℕ

/-- The percentage of the plane enclosed by pentagons -/
def pentagon_percentage (t : PlaneTiling) : ℚ :=
  t.pentagon_squares / (t.grid_size ^ 2 : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 62.5% -/
theorem pentagon_coverage_is_62_5_percent (t : PlaneTiling) 
  (h1 : t.grid_size = 4)
  (h2 : t.pentagon_squares = 10) : 
  pentagon_percentage t = 62.5 := by
  sorry

end pentagon_coverage_is_62_5_percent_l3520_352053


namespace division_invariance_l3520_352059

theorem division_invariance (a b : ℝ) (h : b ≠ 0) : (10 * a) / (10 * b) = a / b := by
  sorry

end division_invariance_l3520_352059


namespace cooking_dishes_time_is_one_point_five_l3520_352069

/-- Represents the daily schedule of a working mom -/
structure DailySchedule where
  total_awake_time : ℝ
  work_time : ℝ
  gym_time : ℝ
  bathing_time : ℝ
  homework_bedtime : ℝ
  packing_lunches : ℝ
  cleaning_time : ℝ
  shower_leisure : ℝ

/-- Calculates the time spent on cooking and dishes -/
def cooking_dishes_time (schedule : DailySchedule) : ℝ :=
  schedule.total_awake_time - (schedule.work_time + schedule.gym_time + 
  schedule.bathing_time + schedule.homework_bedtime + schedule.packing_lunches + 
  schedule.cleaning_time + schedule.shower_leisure)

/-- Theorem stating that the cooking and dishes time for the given schedule is 1.5 hours -/
theorem cooking_dishes_time_is_one_point_five (schedule : DailySchedule) 
  (h1 : schedule.total_awake_time = 16)
  (h2 : schedule.work_time = 8)
  (h3 : schedule.gym_time = 2)
  (h4 : schedule.bathing_time = 0.5)
  (h5 : schedule.homework_bedtime = 1)
  (h6 : schedule.packing_lunches = 0.5)
  (h7 : schedule.cleaning_time = 0.5)
  (h8 : schedule.shower_leisure = 2) :
  cooking_dishes_time schedule = 1.5 := by
  sorry

end cooking_dishes_time_is_one_point_five_l3520_352069


namespace vans_needed_for_field_trip_l3520_352002

theorem vans_needed_for_field_trip (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) :
  van_capacity = 5 → num_students = 25 → num_adults = 5 →
  (num_students + num_adults) / van_capacity = 6 :=
by sorry

end vans_needed_for_field_trip_l3520_352002


namespace remy_water_usage_l3520_352009

/-- Proves that Remy used 25 gallons of water given the conditions of the problem. -/
theorem remy_water_usage (roman : ℕ) (remy : ℕ) : 
  remy = 3 * roman + 1 →  -- Condition 1
  roman + remy = 33 →     -- Condition 2
  remy = 25 := by
sorry

end remy_water_usage_l3520_352009


namespace junior_toys_l3520_352020

theorem junior_toys (rabbits : ℕ) (monday_toys : ℕ) : 
  rabbits = 16 →
  (monday_toys + 2 * monday_toys + 4 * monday_toys + monday_toys) / rabbits = 3 →
  monday_toys = 6 := by
sorry

end junior_toys_l3520_352020


namespace circle_line_intersection_l3520_352030

theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 16 ∧ p.1 = 4 :=
by sorry

end circle_line_intersection_l3520_352030


namespace inequality_solution_set_l3520_352093

theorem inequality_solution_set : 
  {x : ℝ | 8*x^3 + 9*x^2 + 7*x < 6} = 
  {x : ℝ | (-6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)} := by
  sorry

end inequality_solution_set_l3520_352093


namespace truck_driver_earnings_l3520_352089

/-- Calculates the net earnings of a truck driver given specific conditions --/
theorem truck_driver_earnings
  (gas_cost : ℝ)
  (fuel_efficiency : ℝ)
  (driving_speed : ℝ)
  (payment_rate : ℝ)
  (driving_duration : ℝ)
  (h1 : gas_cost = 2)
  (h2 : fuel_efficiency = 10)
  (h3 : driving_speed = 30)
  (h4 : payment_rate = 0.5)
  (h5 : driving_duration = 10)
  : ∃ (net_earnings : ℝ), net_earnings = 90 :=
by
  sorry

end truck_driver_earnings_l3520_352089


namespace min_side_difference_l3520_352007

theorem min_side_difference (a b c : ℕ) : 
  a + b + c = 3010 →
  a < b →
  b ≤ c →
  (∀ x y z : ℕ, x + y + z = 3010 → x < y → y ≤ z → b - a ≤ y - x) →
  b - a = 1 :=
by sorry

end min_side_difference_l3520_352007


namespace sequence_sum_equals_5923_l3520_352055

def arithmetic_sum (a1 l1 d : ℤ) : ℤ :=
  let n := (l1 - a1) / d + 1
  n * (a1 + l1) / 2

def sequence_sum : ℤ :=
  3 * (arithmetic_sum 45 93 2) + 2 * (arithmetic_sum (-4) 38 2)

theorem sequence_sum_equals_5923 : sequence_sum = 5923 := by
  sorry

end sequence_sum_equals_5923_l3520_352055


namespace power_of_product_l3520_352051

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := by
  sorry

end power_of_product_l3520_352051


namespace more_than_half_inside_l3520_352072

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The circle is inscribed in the triangle -/
  inscribed : circle ⊆ triangle

/-- A square circumscribed around a circle -/
structure CircumscribedSquare where
  /-- The square -/
  square : Set (ℝ × ℝ)
  /-- The circumscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- The square is circumscribed around the circle -/
  circumscribed : circle ⊆ square

/-- The perimeter of a square -/
def squarePerimeter (s : CircumscribedSquare) : ℝ := sorry

/-- The length of the square's perimeter segments inside the triangle -/
def insidePerimeterLength (t : InscribedTriangle) (s : CircumscribedSquare) : ℝ := sorry

/-- Main theorem: More than half of the square's perimeter is inside the triangle -/
theorem more_than_half_inside (t : InscribedTriangle) (s : CircumscribedSquare) 
  (h : t.circle = s.circle) : 
  insidePerimeterLength t s > squarePerimeter s / 2 := by sorry

end more_than_half_inside_l3520_352072


namespace max_value_theorem_l3520_352011

theorem max_value_theorem (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_sum : a + b + c + d ≤ 4) : 
  (a^2 * (a + b))^(1/4) + (b^2 * (b + c))^(1/4) + 
  (c^2 * (c + d))^(1/4) + (d^2 * (d + a))^(1/4) ≤ 4 * 2^(1/4) := by
  sorry

end max_value_theorem_l3520_352011


namespace rectangles_in_35_44_grid_l3520_352070

/-- The number of rectangles in a grid -/
def count_rectangles (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Theorem: The number of rectangles in a 35 · 44 grid is 87 -/
theorem rectangles_in_35_44_grid :
  count_rectangles 35 44 = 87 := by
  sorry

end rectangles_in_35_44_grid_l3520_352070


namespace angle_BDC_value_l3520_352021

-- Define the angles in degrees
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Define the theorem
theorem angle_BDC_value :
  ∀ (angle_BDC : ℝ),
  -- ABC is a straight line (implied by the exterior angle theorem)
  angle_ABD = angle_BCD + angle_BDC →
  angle_BDC = 36 := by
sorry

end angle_BDC_value_l3520_352021


namespace unique_digit_product_equation_l3520_352087

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_digit_product_equation : 
  ∃! x : ℕ, digit_product x = x^2 - 10*x - 22 ∧ x = 12 := by
sorry

end unique_digit_product_equation_l3520_352087


namespace petting_zoo_count_l3520_352027

/-- The number of animals Mary counted -/
def mary_count : ℕ := 130

/-- The number of animals Mary double-counted -/
def double_counted : ℕ := 19

/-- The number of animals Mary missed -/
def missed : ℕ := 39

/-- The actual number of animals in the petting zoo -/
def actual_count : ℕ := 150

theorem petting_zoo_count : 
  mary_count - double_counted + missed = actual_count := by sorry

end petting_zoo_count_l3520_352027


namespace base4_division_theorem_l3520_352084

/-- Represents a number in base 4 --/
def Base4 : Type := Nat

/-- Converts a Base4 number to its decimal representation --/
def to_decimal (n : Base4) : Nat :=
  sorry

/-- Converts a decimal number to its Base4 representation --/
def to_base4 (n : Nat) : Base4 :=
  sorry

/-- Performs division in Base4 --/
def base4_div (a b : Base4) : Base4 :=
  sorry

theorem base4_division_theorem :
  base4_div (to_base4 1023) (to_base4 11) = to_base4 33 := by
  sorry

end base4_division_theorem_l3520_352084


namespace log_product_range_l3520_352035

theorem log_product_range :
  let y := Real.log 6 / Real.log 5 *
           Real.log 7 / Real.log 6 *
           Real.log 8 / Real.log 7 *
           Real.log 9 / Real.log 8 *
           Real.log 10 / Real.log 9
  1 < y ∧ y < 2 := by sorry

end log_product_range_l3520_352035


namespace tens_digit_of_2023_pow_2024_minus_2025_l3520_352057

theorem tens_digit_of_2023_pow_2024_minus_2025 : ∃ k : ℕ, (2023^2024 - 2025) % 100 = 10 * k + 6 := by
  sorry

end tens_digit_of_2023_pow_2024_minus_2025_l3520_352057


namespace base_76_congruence_l3520_352004

theorem base_76_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 18) 
  (h3 : (276935824 : ℤ) ≡ b [ZMOD 17]) : b = 0 ∨ b = 17 := by
  sorry

#check base_76_congruence

end base_76_congruence_l3520_352004


namespace quadratic_function_property_l3520_352032

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) :=
by sorry

end quadratic_function_property_l3520_352032


namespace salary_proof_l3520_352049

/-- The weekly salary of employee N -/
def N_salary : ℝ := 275

/-- The weekly salary of employee M -/
def M_salary (N_salary : ℝ) : ℝ := 1.2 * N_salary

/-- The total weekly salary for both employees -/
def total_salary : ℝ := 605

theorem salary_proof :
  N_salary + M_salary N_salary = total_salary :=
sorry

end salary_proof_l3520_352049


namespace johnny_take_home_pay_is_67_32_l3520_352054

/-- Calculates Johnny's take-home pay after taxes based on his work hours and pay rates. -/
def johnny_take_home_pay (task_a_rate : ℝ) (task_b_rate : ℝ) (total_hours : ℝ) (task_a_hours : ℝ) (tax_rate : ℝ) : ℝ :=
  let task_b_hours := total_hours - task_a_hours
  let total_earnings := task_a_rate * task_a_hours + task_b_rate * task_b_hours
  let tax := tax_rate * total_earnings
  total_earnings - tax

/-- Proves that Johnny's take-home pay after taxes is $67.32 given the specified conditions. -/
theorem johnny_take_home_pay_is_67_32 :
  johnny_take_home_pay 6.75 8.25 10 4 0.12 = 67.32 := by
  sorry

end johnny_take_home_pay_is_67_32_l3520_352054


namespace remainder_after_adding_1470_l3520_352068

theorem remainder_after_adding_1470 (n : ℤ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := by
  sorry

end remainder_after_adding_1470_l3520_352068


namespace james_and_louise_ages_james_and_louise_ages_proof_l3520_352095

theorem james_and_louise_ages : ℕ → ℕ → Prop :=
  fun j l =>
    (j = l + 9) →                  -- James is nine years older than Louise
    (j + 7 = 3 * (l - 3)) →        -- Seven years from now, James will be three times as old as Louise was three years before now
    (j + l = 35)                   -- The sum of their current ages is 35

-- The proof of this theorem
theorem james_and_louise_ages_proof : ∃ j l : ℕ, james_and_louise_ages j l := by
  sorry

end james_and_louise_ages_james_and_louise_ages_proof_l3520_352095


namespace expression_evaluation_l3520_352073

theorem expression_evaluation (x y : ℚ) (hx : x = -1) (hy : y = -1/3) :
  (3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y) = 8 := by
sorry

end expression_evaluation_l3520_352073


namespace managers_salary_l3520_352074

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 15 →
  avg_salary = 1800 →
  avg_increase = 150 →
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary = 4200 := by
  sorry

end managers_salary_l3520_352074


namespace houses_with_neither_l3520_352067

theorem houses_with_neither (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ)
  (h_total : total = 70)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_both : both = 35) :
  total - (garage + pool - both) = 15 :=
by sorry

end houses_with_neither_l3520_352067


namespace integer_sum_problem_l3520_352031

theorem integer_sum_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 120) : 
  x.val + y.val = 2 * Real.sqrt 136 := by
sorry

end integer_sum_problem_l3520_352031


namespace rahul_deepak_age_ratio_l3520_352010

/-- Proves that the ratio of Rahul's age to Deepak's age is 4:3 -/
theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
  deepak_age = 12 →
  rahul_age + 10 = 26 →
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 :=
by
  sorry

end rahul_deepak_age_ratio_l3520_352010


namespace toy_cost_price_l3520_352061

theorem toy_cost_price (profit_equality : 30 * (12 - C) = 20 * (15 - C)) : C = 6 :=
by sorry

end toy_cost_price_l3520_352061


namespace intersection_A_B_union_complement_A_B_l3520_352045

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (Aᶜ) ∪ B = {x : ℝ | 0 < x} := by sorry

end intersection_A_B_union_complement_A_B_l3520_352045


namespace consecutive_episodes_probability_l3520_352022

theorem consecutive_episodes_probability (n : ℕ) (h : n = 6) :
  let total_combinations := n.choose 2
  let consecutive_pairs := n - 1
  (consecutive_pairs : ℚ) / total_combinations = 1 / 3 := by
sorry

end consecutive_episodes_probability_l3520_352022


namespace arithmetic_sequence_formula_l3520_352085

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n+1) - a n = d) 
  (h2 : a 1 = f (d - 1)) 
  (h3 : a 3 = f (d + 1)) :
  ∀ n, a n = 2*n + 1 :=
sorry

end arithmetic_sequence_formula_l3520_352085


namespace inequality_solution_range_l3520_352028

theorem inequality_solution_range (k : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + k*x - 1 > 0) → k > -3/2 :=
by sorry

end inequality_solution_range_l3520_352028


namespace no_periodic_sum_with_periods_2_and_pi_div_2_l3520_352043

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- The period of a function is a positive real number p such that f(x + p) = f(x) for all x. -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem no_periodic_sum_with_periods_2_and_pi_div_2 :
  ¬ ∃ (g h : ℝ → ℝ),
    Periodic g ∧ Periodic h ∧ IsPeriod g 2 ∧ IsPeriod h (π / 2) ∧ Periodic (g + h) :=
sorry

end no_periodic_sum_with_periods_2_and_pi_div_2_l3520_352043


namespace square_equality_l3520_352088

theorem square_equality (n : ℕ) : (n + 3)^2 = 3*(n + 2)^2 - 3*(n + 1)^2 + n^2 := by
  sorry

end square_equality_l3520_352088


namespace factorization_equality_l3520_352017

theorem factorization_equality (a x y : ℝ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l3520_352017


namespace tournament_players_count_l3520_352080

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players not in the lowest 8
  total_players : ℕ := n + 8
  points_among_n : ℕ := n * (n - 1) / 2
  points_n_vs_lowest8 : ℕ := points_among_n / 3
  points_among_lowest8 : ℕ := 28
  total_points : ℕ := 4 * points_among_n / 3 + 2 * points_among_lowest8

/-- The theorem stating that the total number of players in the tournament is 50 -/
theorem tournament_players_count (t : Tournament) : t.total_players = 50 := by
  sorry

end tournament_players_count_l3520_352080


namespace sector_area_from_arc_length_l3520_352033

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) (h : 2 * r = 4) : 
  (1 / 2) * r^2 * 2 = 4 := by
  sorry

end sector_area_from_arc_length_l3520_352033


namespace prob_at_least_one_girl_l3520_352063

/-- The probability of selecting at least one girl from a group of 4 boys and 3 girls when choosing 2 people -/
theorem prob_at_least_one_girl (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 4 → num_girls = 3 → 
  (1 - (Nat.choose num_boys 2 : ℚ) / (Nat.choose (num_boys + num_girls) 2 : ℚ)) = 5/7 :=
by sorry

end prob_at_least_one_girl_l3520_352063
