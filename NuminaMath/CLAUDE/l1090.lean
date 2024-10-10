import Mathlib

namespace carolyn_initial_marbles_l1090_109066

/-- Represents the number of marbles Carolyn started with -/
def initial_marbles : ℕ := sorry

/-- Represents the number of items Carolyn shared with Diana -/
def shared_items : ℕ := 42

/-- Represents the number of marbles Carolyn ended with -/
def remaining_marbles : ℕ := 5

/-- Theorem stating that Carolyn started with 47 marbles -/
theorem carolyn_initial_marbles : initial_marbles = 47 := by
  sorry

end carolyn_initial_marbles_l1090_109066


namespace front_page_stickers_l1090_109036

/-- Given:
  * initial_stickers: The initial number of stickers Mary had
  * pages: The number of pages (excluding the front page) where Mary used stickers
  * stickers_per_page: The number of stickers Mary used on each page (excluding the front page)
  * remaining_stickers: The number of stickers Mary has left
  
  Prove that the number of large stickers used on the front page is 3
-/
theorem front_page_stickers 
  (initial_stickers : ℕ) 
  (pages : ℕ) 
  (stickers_per_page : ℕ) 
  (remaining_stickers : ℕ) 
  (h1 : initial_stickers = 89)
  (h2 : pages = 6)
  (h3 : stickers_per_page = 7)
  (h4 : remaining_stickers = 44) :
  initial_stickers - (pages * stickers_per_page) - remaining_stickers = 3 :=
by sorry

end front_page_stickers_l1090_109036


namespace supercomputer_multiplications_l1090_109050

/-- The number of multiplications a supercomputer can perform in half a day -/
def multiplications_in_half_day (multiplications_per_second : ℕ) : ℕ :=
  multiplications_per_second * (12 * 3600)

/-- Theorem stating that a supercomputer performing 80,000 multiplications per second
    will execute 3,456,000,000 multiplications in half a day -/
theorem supercomputer_multiplications :
  multiplications_in_half_day 80000 = 3456000000 := by
  sorry

#eval multiplications_in_half_day 80000

end supercomputer_multiplications_l1090_109050


namespace geometric_series_sum_is_15_11_l1090_109048

/-- The sum of an infinite geometric series with first term a and common ratio r -/
def geometricSeriesSum (a : ℚ) (r : ℚ) : ℚ := a / (1 - r)

/-- The first term of the given geometric series -/
def a : ℚ := 5 / 3

/-- The common ratio of the given geometric series -/
def r : ℚ := -2 / 9

/-- The theorem stating that the sum of the given infinite geometric series is 15/11 -/
theorem geometric_series_sum_is_15_11 : geometricSeriesSum a r = 15 / 11 := by sorry

end geometric_series_sum_is_15_11_l1090_109048


namespace complex_equation_solution_l1090_109008

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : z / (2 + a * I) = 2 / (1 + I)) 
  (h2 : z.im = -3) : 
  z.re = 1 := by
  sorry

end complex_equation_solution_l1090_109008


namespace distance_after_walk_l1090_109082

/-- A regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  regular : side_length = 3

/-- Walking distance along the perimeter -/
def walking_distance : ℝ := 10

/-- Calculate the distance between start and end points after walking along the perimeter -/
def distance_from_start (h : RegularHexagon) (d : ℝ) : ℝ := sorry

/-- Theorem: The distance from start to end after walking 10 km on a regular hexagon with 3 km sides is 1 km -/
theorem distance_after_walk (h : RegularHexagon) :
  distance_from_start h walking_distance = 1 := by sorry

end distance_after_walk_l1090_109082


namespace max_log_product_l1090_109083

theorem max_log_product (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  ∃ (max_val : ℝ), max_val = 4 ∧ ∀ (a b : ℝ), a > 1 → b > 1 → Real.log a / Real.log 10 + Real.log b / Real.log 10 = 4 →
    Real.log x / Real.log 10 * Real.log y / Real.log 10 ≥ Real.log a / Real.log 10 * Real.log b / Real.log 10 :=
by
  sorry

#check max_log_product

end max_log_product_l1090_109083


namespace milly_boas_count_l1090_109041

theorem milly_boas_count (tail_feathers_per_flamingo : ℕ) 
                         (safe_pluck_percentage : ℚ)
                         (feathers_per_boa : ℕ)
                         (flamingoes_to_harvest : ℕ) :
  tail_feathers_per_flamingo = 20 →
  safe_pluck_percentage = 1/4 →
  feathers_per_boa = 200 →
  flamingoes_to_harvest = 480 →
  (↑flamingoes_to_harvest * safe_pluck_percentage * ↑tail_feathers_per_flamingo) / ↑feathers_per_boa = 12 :=
by sorry

end milly_boas_count_l1090_109041


namespace decimal_72_to_octal_l1090_109055

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem decimal_72_to_octal :
  decimal_to_octal 72 = [1, 1, 0] := by sorry

end decimal_72_to_octal_l1090_109055


namespace tanner_remaining_money_l1090_109037

def september_savings : ℕ := 17
def october_savings : ℕ := 48
def november_savings : ℕ := 25
def video_game_cost : ℕ := 49

theorem tanner_remaining_money :
  september_savings + october_savings + november_savings - video_game_cost = 41 := by
  sorry

end tanner_remaining_money_l1090_109037


namespace binomial_product_l1090_109023

theorem binomial_product (x : ℝ) : (5 * x - 3) * (2 * x + 4) = 10 * x^2 + 14 * x - 12 := by
  sorry

end binomial_product_l1090_109023


namespace probability_red_or_white_l1090_109038

/-- Probability of selecting a red or white marble from a bag -/
theorem probability_red_or_white (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h_total : total = 30)
  (h_blue : blue = 5)
  (h_red : red = 9)
  (h_positive : total > 0) :
  (red + (total - blue - red)) / total = 5 / 6 := by
  sorry

end probability_red_or_white_l1090_109038


namespace tree_spacing_l1090_109003

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first and last tree
    is 175 feet. -/
theorem tree_spacing (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  (n - 1) * d / 4 = 175 := by
  sorry

end tree_spacing_l1090_109003


namespace students_below_50_l1090_109010

/-- Represents the frequency distribution of scores -/
structure ScoreDistribution where
  freq_50_60 : Real
  freq_60_70 : Real
  freq_70_80 : Real
  freq_80_90 : Real
  freq_90_100 : Real

/-- The problem statement -/
theorem students_below_50 
  (total_students : Nat) 
  (selected_students : Nat)
  (score_distribution : ScoreDistribution)
  (h1 : total_students = 600)
  (h2 : selected_students = 60)
  (h3 : score_distribution.freq_50_60 = 0.15)
  (h4 : score_distribution.freq_60_70 = 0.15)
  (h5 : score_distribution.freq_70_80 = 0.30)
  (h6 : score_distribution.freq_80_90 = 0.25)
  (h7 : score_distribution.freq_90_100 = 0.05) :
  (total_students : Real) * (1 - (score_distribution.freq_50_60 + 
                                  score_distribution.freq_60_70 + 
                                  score_distribution.freq_70_80 + 
                                  score_distribution.freq_80_90 + 
                                  score_distribution.freq_90_100)) = 60 := by
  sorry

end students_below_50_l1090_109010


namespace jasons_books_l1090_109021

theorem jasons_books (keith_books : ℕ) (total_books : ℕ) (h1 : keith_books = 20) (h2 : total_books = 41) :
  total_books - keith_books = 21 := by
sorry

end jasons_books_l1090_109021


namespace fifth_term_of_sequence_l1090_109011

def second_order_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ → ℕ, (∀ n, d (n + 1) = d n + 2) ∧
               (∀ n, a (n + 1) = a n + d n)

theorem fifth_term_of_sequence
  (a : ℕ → ℕ)
  (h : second_order_arithmetic_sequence a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 7)
  (h4 : a 4 = 13) :
  a 5 = 21 := by
  sorry

end fifth_term_of_sequence_l1090_109011


namespace complement_of_A_l1090_109026

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x < 0}

-- State the theorem
theorem complement_of_A (x : ℝ) : x ∈ (Set.compl A) ↔ 0 ≤ x ∧ x < 1 := by
  sorry

end complement_of_A_l1090_109026


namespace parallel_vectors_m_value_l1090_109033

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (2, m)
  parallel a b → m = -4 := by
  sorry

end parallel_vectors_m_value_l1090_109033


namespace jason_gave_keith_47_pears_l1090_109029

/-- The number of pears Jason gave to Keith -/
def pears_given_to_keith (initial_pears : ℕ) (pears_from_mike : ℕ) (pears_left : ℕ) : ℕ :=
  initial_pears + pears_from_mike - pears_left

theorem jason_gave_keith_47_pears :
  pears_given_to_keith 46 12 11 = 47 := by
  sorry

end jason_gave_keith_47_pears_l1090_109029


namespace restaurant_bill_proof_l1090_109070

/-- Calculates the total bill given the number of people and the amount each person paid -/
def totalBill (numPeople : ℕ) (amountPerPerson : ℕ) : ℕ :=
  numPeople * amountPerPerson

/-- Proves that if three people divide a bill evenly and each pays $45, then the total bill is $135 -/
theorem restaurant_bill_proof :
  totalBill 3 45 = 135 := by
  sorry

end restaurant_bill_proof_l1090_109070


namespace average_height_is_64_inches_l1090_109000

/-- Given the heights of Parker, Daisy, and Reese, prove their average height is 64 inches. -/
theorem average_height_is_64_inches 
  (reese_height : ℕ)
  (daisy_height : ℕ)
  (parker_height : ℕ)
  (h1 : reese_height = 60)
  (h2 : daisy_height = reese_height + 8)
  (h3 : parker_height = daisy_height - 4) :
  (reese_height + daisy_height + parker_height) / 3 = 64 := by
  sorry

end average_height_is_64_inches_l1090_109000


namespace find_N_l1090_109022

theorem find_N : ∃! (N : ℕ), N > 0 ∧ 18^2 * 45^2 = 15^2 * N^2 ∧ N = 81 := by sorry

end find_N_l1090_109022


namespace fifteenth_student_age_l1090_109009

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15) 
  (h2 : avg_age_all = 15) 
  (h3 : group1_size = 7) 
  (h4 : avg_age_group1 = 14) 
  (h5 : group2_size = 7) 
  (h6 : avg_age_group2 = 16) : 
  ℝ := by
  sorry

#check fifteenth_student_age

end fifteenth_student_age_l1090_109009


namespace lian_lian_sales_properties_l1090_109072

/-- Represents the sales data and growth rate for the "Lian Lian" store -/
structure SalesData where
  june_sales : ℕ
  august_sales : ℕ
  months_between : ℕ
  growth_rate : ℝ

/-- Calculates the projected sales for the next month -/
def project_next_month_sales (data : SalesData) : ℝ :=
  data.august_sales * (1 + data.growth_rate)

/-- Theorem stating the properties of the "Lian Lian" store's sales data -/
theorem lian_lian_sales_properties (data : SalesData) 
  (h1 : data.june_sales = 30000)
  (h2 : data.august_sales = 36300)
  (h3 : data.months_between = 2)
  (h4 : data.growth_rate = (Real.sqrt 1.21 - 1)) :
  data.growth_rate = 0.1 ∧ project_next_month_sales data < 40000 := by
  sorry

#check lian_lian_sales_properties

end lian_lian_sales_properties_l1090_109072


namespace square_side_length_l1090_109051

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
  sorry

end square_side_length_l1090_109051


namespace van_distance_proof_l1090_109016

/-- Proves that the distance covered by a van is 180 km given specific conditions -/
theorem van_distance_proof (D : ℝ) (original_time : ℝ) (new_time : ℝ) (new_speed : ℝ) :
  original_time = 6 →
  new_time = 3/2 * original_time →
  new_speed = 20 →
  D = new_speed * new_time →
  D = 180 := by
  sorry

#check van_distance_proof

end van_distance_proof_l1090_109016


namespace spiders_in_playground_sami_spiders_l1090_109024

theorem spiders_in_playground (ants : ℕ) (initial_ladybugs : ℕ) (departed_ladybugs : ℕ) (total_insects : ℕ) : ℕ :=
  by
  sorry

-- Definitions and conditions
def ants : ℕ := 12
def initial_ladybugs : ℕ := 8
def departed_ladybugs : ℕ := 2
def total_insects : ℕ := 21

-- Theorem statement
theorem sami_spiders : spiders_in_playground ants initial_ladybugs departed_ladybugs total_insects = 3 := by
  sorry

end spiders_in_playground_sami_spiders_l1090_109024


namespace enrollment_change_l1090_109097

theorem enrollment_change (E : ℝ) (E_1992 E_1993 E_1994 E_1995 : ℝ) : 
  E_1992 = 1.20 * E →
  E_1993 = 1.15 * E_1992 →
  E_1994 = 0.90 * E_1993 →
  E_1995 = 1.25 * E_1994 →
  (E_1995 - E) / E = 0.5525 := by
  sorry

end enrollment_change_l1090_109097


namespace incorrect_logical_statement_l1090_109053

theorem incorrect_logical_statement : 
  ¬(∀ (p q : Prop), (¬p ∨ ¬q) → (¬p ∧ ¬q)) := by
  sorry

end incorrect_logical_statement_l1090_109053


namespace a_steps_equals_b_steps_l1090_109088

/-- The number of steps in a stationary escalator -/
def stationary_steps : ℕ := 100

/-- The number of steps B takes -/
def b_steps : ℕ := 75

/-- The relative speed of A compared to B -/
def relative_speed : ℚ := 1/3

theorem a_steps_equals_b_steps :
  ∃ (a : ℕ) (e : ℚ),
    -- A's steps plus escalator movement equals total steps
    a + e = stationary_steps ∧
    -- B's steps plus escalator movement equals total steps
    b_steps + e = stationary_steps ∧
    -- A's speed is 1/3 of B's speed
    a * relative_speed = b_steps * (1 : ℚ) →
    a = b_steps :=
by sorry

end a_steps_equals_b_steps_l1090_109088


namespace min_value_theorem_l1090_109001

/-- Given positive real numbers a and b, this theorem proves that the minimum value of
    (a + 2/b)(a + 2/b - 1010) + (b + 2/a)(b + 2/a - 1010) + 101010 is -404040. -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 2/y) * (x + 2/y - 1010) + (y + 2/x) * (y + 2/x - 1010) + 101010 <
  (a + 2/b) * (a + 2/b - 1010) + (b + 2/a) * (b + 2/a - 1010) + 101010) →
  (a + 2/b) * (a + 2/b - 1010) + (b + 2/a) * (b + 2/a - 1010) + 101010 ≥ -404040 :=
by sorry

end min_value_theorem_l1090_109001


namespace sqrt_54_times_sqrt_one_third_l1090_109045

theorem sqrt_54_times_sqrt_one_third : Real.sqrt 54 * Real.sqrt (1/3) = 3 * Real.sqrt 2 := by
  sorry

end sqrt_54_times_sqrt_one_third_l1090_109045


namespace john_current_age_l1090_109012

/-- John's current age -/
def john_age : ℕ := sorry

/-- John's sister's current age -/
def sister_age : ℕ := sorry

/-- John's sister is twice his age -/
axiom sister_twice_age : sister_age = 2 * john_age

/-- When John is 50, his sister will be 60 -/
axiom future_ages : sister_age + (50 - john_age) = 60

theorem john_current_age : john_age = 10 := by sorry

end john_current_age_l1090_109012


namespace total_area_approx_33_87_l1090_109014

/-- Converts feet and inches to meters -/
def to_meters (feet : ℕ) (inches : ℕ) : ℝ :=
  feet * 0.3048 + inches * 0.0254

/-- Calculates the area of a room in square meters -/
def room_area (length_feet : ℕ) (length_inches : ℕ) (width_feet : ℕ) (width_inches : ℕ) : ℝ :=
  to_meters length_feet length_inches * to_meters width_feet width_inches

/-- Theorem: The total area of three rooms is approximately 33.87 square meters -/
theorem total_area_approx_33_87 :
  let room_a := room_area 14 8 10 5
  let room_b := room_area 12 3 11 2
  let room_c := room_area 9 7 7 10
  let total_area := room_a + room_b + room_c
  ∃ ε > 0, |total_area - 33.87| < ε :=
by
  sorry


end total_area_approx_33_87_l1090_109014


namespace ice_cream_cost_l1090_109043

/-- The cost of ice cream problem -/
theorem ice_cream_cost (ice_cream_quantity : ℕ) (yoghurt_quantity : ℕ) (yoghurt_cost : ℚ) (price_difference : ℚ) :
  ice_cream_quantity = 10 →
  yoghurt_quantity = 4 →
  yoghurt_cost = 1 →
  price_difference = 36 →
  ∃ (ice_cream_cost : ℚ),
    ice_cream_cost * ice_cream_quantity = yoghurt_cost * yoghurt_quantity + price_difference ∧
    ice_cream_cost = 4 :=
by sorry

end ice_cream_cost_l1090_109043


namespace sixth_term_of_sequence_l1090_109054

theorem sixth_term_of_sequence (a : ℕ → ℕ) (h : ∀ n, a n = 2 * n + 1) : a 6 = 13 := by
  sorry

end sixth_term_of_sequence_l1090_109054


namespace xy_power_equality_l1090_109061

theorem xy_power_equality (x y : ℕ) (h : x ≠ y) :
  x ^ y = y ^ x ↔ (x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2) := by
  sorry

end xy_power_equality_l1090_109061


namespace max_grain_mass_on_platform_l1090_109060

/-- Represents a rectangular platform with grain piled on it. -/
structure GrainPlatform where
  length : ℝ
  width : ℝ
  grainDensity : ℝ
  maxAngle : ℝ

/-- Calculates the maximum mass of grain on the platform. -/
def maxGrainMass (platform : GrainPlatform) : ℝ :=
  sorry

/-- Theorem stating the maximum mass of grain on the given platform. -/
theorem max_grain_mass_on_platform :
  let platform : GrainPlatform := {
    length := 8,
    width := 5,
    grainDensity := 1200,
    maxAngle := π / 4  -- 45 degrees in radians
  }
  maxGrainMass platform = 47500  -- 47.5 tons in kg
  := by sorry

end max_grain_mass_on_platform_l1090_109060


namespace ratio_p_to_q_l1090_109069

theorem ratio_p_to_q (p q : ℚ) (h : 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : p / q = 29 / 10 := by
  sorry

end ratio_p_to_q_l1090_109069


namespace quadratic_equation_solution_l1090_109046

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - x = 0 ↔ x = 0 ∨ x = 1 := by sorry

end quadratic_equation_solution_l1090_109046


namespace area_of_polygon_ABHFGD_l1090_109013

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the squares
def is_square (P Q R S : ℝ × ℝ) : Prop := sorry

-- Define the area of a polygon
def area (points : List (ℝ × ℝ)) : ℝ := sorry

-- Define the midpoint of a line segment
def is_midpoint (M P Q : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem area_of_polygon_ABHFGD :
  is_square A B C D →
  is_square E F G D →
  area [A, B, C, D] = 36 →
  area [E, F, G, D] = 36 →
  is_midpoint H B C →
  is_midpoint H E F →
  area [A, B, H, F, G, D] = 36 := sorry

end area_of_polygon_ABHFGD_l1090_109013


namespace deceased_cannot_marry_l1090_109079

-- Define a person
structure Person where
  alive : Bool

-- Define marriage as a relation between two people
def canMarry (p1 p2 : Person) : Prop := p1.alive ∧ p2.alive

-- Theorem: A deceased person cannot marry anyone
theorem deceased_cannot_marry (p1 p2 : Person) : 
  ¬p1.alive → ¬(canMarry p1 p2) := by
  sorry

#check deceased_cannot_marry

end deceased_cannot_marry_l1090_109079


namespace power_zero_eq_one_l1090_109071

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_eq_one_l1090_109071


namespace final_balance_l1090_109089

def bank_account_balance (initial_savings withdrawal deposit : ℕ) : ℕ :=
  initial_savings - withdrawal + deposit

theorem final_balance (initial_savings withdrawal : ℕ) 
  (h1 : initial_savings = 230)
  (h2 : withdrawal = 60)
  (h3 : bank_account_balance initial_savings withdrawal (2 * withdrawal) = 290) : 
  bank_account_balance initial_savings withdrawal (2 * withdrawal) = 290 := by
  sorry

end final_balance_l1090_109089


namespace solve_equation_l1090_109058

theorem solve_equation (c m n x : ℝ) 
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hmn : m ≠ n)
  (hc : c = 3)
  (hm2 : m = 2)
  (hn5 : n = 5)
  (heq : (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) :
  x = -11 := by
  sorry

end solve_equation_l1090_109058


namespace cookies_to_sell_l1090_109084

theorem cookies_to_sell (total : ℕ) (grandmother : ℕ) (uncle : ℕ) (neighbor : ℕ) 
  (h1 : total = 50)
  (h2 : grandmother = 12)
  (h3 : uncle = 7)
  (h4 : neighbor = 5) :
  total - (grandmother + uncle + neighbor) = 26 := by
  sorry

end cookies_to_sell_l1090_109084


namespace quadratic_equation_solution_l1090_109078

theorem quadratic_equation_solution : 
  let x₁ : ℝ := 2 + Real.sqrt 11
  let x₂ : ℝ := 2 - Real.sqrt 11
  ∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ (x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l1090_109078


namespace xy_value_l1090_109005

theorem xy_value (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 := by
  sorry

end xy_value_l1090_109005


namespace inequality_solution_set_l1090_109007

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 - |x| + 6 > 0

-- Define the solution set
def solution_set : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
sorry

end inequality_solution_set_l1090_109007


namespace smallest_repetition_for_divisibility_l1090_109031

/-- The sum of digits of 2013 -/
def sum_digits_2013 : ℕ := 2 + 0 + 1 + 3

/-- Function to check if a number is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

/-- The concatenated number when 2013 is repeated n times -/
def repeated_2013 (n : ℕ) : ℕ := 
  let digits := sum_digits_2013 * n
  digits

/-- The smallest positive integer n such that the number formed by 
    concatenating 2013 n times is divisible by 9 is equal to 3 -/
theorem smallest_repetition_for_divisibility : 
  (∃ n : ℕ, n > 0 ∧ is_divisible_by_9 (repeated_2013 n)) ∧ 
  (∀ k : ℕ, k > 0 → is_divisible_by_9 (repeated_2013 k) → k ≥ 3) ∧
  is_divisible_by_9 (repeated_2013 3) :=
sorry

end smallest_repetition_for_divisibility_l1090_109031


namespace system_solution_inequality_solution_l1090_109067

-- Define the system of equations
def system_eq (x y : ℝ) : Prop :=
  5 * x + 2 * y = 25 ∧ 3 * x + 4 * y = 15

-- Define the linear inequality
def linear_ineq (x : ℝ) : Prop :=
  2 * x - 6 < 3 * x

-- Theorem for the system of equations
theorem system_solution :
  ∃! (x y : ℝ), system_eq x y ∧ x = 5 ∧ y = 0 :=
sorry

-- Theorem for the linear inequality
theorem inequality_solution :
  ∀ x : ℝ, linear_ineq x ↔ x > -6 :=
sorry

end system_solution_inequality_solution_l1090_109067


namespace real_roots_iff_m_leq_quarter_m_eq_neg_one_when_sum_and_product_condition_l1090_109049

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2*m - 1)^2 - 4*1*m^2

-- Theorem for the range of m for real roots
theorem real_roots_iff_m_leq_quarter (m : ℝ) :
  (∃ x : ℝ, quadratic m x = 0) ↔ m ≤ 1/4 := by sorry

-- Theorem for the value of m when x₁x₂ + x₁ + x₂ = 4
theorem m_eq_neg_one_when_sum_and_product_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁*x₂ + x₁ + x₂ = 4) →
  m = -1 := by sorry

end real_roots_iff_m_leq_quarter_m_eq_neg_one_when_sum_and_product_condition_l1090_109049


namespace base_k_representation_of_fraction_l1090_109047

/-- The base of the number system -/
def k : ℕ+ := sorry

/-- The fraction we're representing -/
def fraction : ℚ := 11 / 85

/-- The repeating part of the base-k representation -/
def repeating_part : ℕ × ℕ := (3, 5)

/-- The value of the repeating base-k representation -/
def repeating_value (k : ℕ+) (rep : ℕ × ℕ) : ℚ :=
  (rep.1 : ℚ) / (k : ℚ) + (rep.2 : ℚ) / ((k : ℚ) ^ 2) 
  / (1 - 1 / ((k : ℚ) ^ 2))

/-- The main theorem -/
theorem base_k_representation_of_fraction :
  repeating_value k repeating_part = fraction ∧ k = 25 := by sorry

end base_k_representation_of_fraction_l1090_109047


namespace income_comparison_l1090_109094

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : mary = 1.5 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mary = 0.9 * juan := by
sorry

end income_comparison_l1090_109094


namespace function_property_l1090_109030

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_property (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end function_property_l1090_109030


namespace solution_of_equation_l1090_109064

theorem solution_of_equation : ∃ x : ℝ, (3 / (x - 2) = 1) ∧ (x = 5) := by
  sorry

end solution_of_equation_l1090_109064


namespace cryptarithmetic_puzzle_l1090_109073

theorem cryptarithmetic_puzzle :
  ∀ (E F D : ℕ),
    E + F + D = E * F - 3 →
    E - F = 2 →
    E ≠ F ∧ E ≠ D ∧ F ≠ D →
    E ≤ 9 ∧ F ≤ 9 ∧ D ≤ 9 →
    D = 4 := by
  sorry

end cryptarithmetic_puzzle_l1090_109073


namespace martians_cannot_hold_hands_l1090_109015

/-- Represents the number of hands a Martian has -/
def martian_hands : ℕ := 3

/-- Represents the number of Martians -/
def num_martians : ℕ := 7

/-- Calculates the total number of hands for all Martians -/
def total_hands : ℕ := martian_hands * num_martians

/-- Theorem stating that seven Martians cannot hold hands with each other -/
theorem martians_cannot_hold_hands : ¬ (total_hands % 2 = 0) := by
  sorry

end martians_cannot_hold_hands_l1090_109015


namespace third_runner_distance_l1090_109087

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ → ℝ

/-- The race setup -/
structure Race where
  length : ℝ
  runner1 : Runner
  runner2 : Runner
  runner3 : Runner

/-- Properties of the race -/
def RaceProperties (race : Race) : Prop :=
  -- The race is 100 meters long
  race.length = 100 ∧
  -- Runner1 is fastest, followed by runner2, then runner3
  race.runner1.speed > race.runner2.speed ∧ race.runner2.speed > race.runner3.speed ∧
  -- All runners maintain constant speeds
  (∀ t : ℝ, race.runner1.distance t = race.runner1.speed * t) ∧
  (∀ t : ℝ, race.runner2.distance t = race.runner2.speed * t) ∧
  (∀ t : ℝ, race.runner3.distance t = race.runner3.speed * t) ∧
  -- When runner1 finishes, runner2 is 10m behind
  race.runner2.distance (race.length / race.runner1.speed) = race.length - 10 ∧
  -- When runner2 finishes, runner3 is 10m behind
  race.runner3.distance (race.length / race.runner2.speed) = race.length - 10

/-- The main theorem to prove -/
theorem third_runner_distance (race : Race) (h : RaceProperties race) :
  race.runner3.distance (race.length / race.runner1.speed) = race.length - 19 := by
  sorry

end third_runner_distance_l1090_109087


namespace units_digit_of_5_to_10_l1090_109081

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_5_to_10 : unitsDigit (5^10) = 5 := by
  sorry

end units_digit_of_5_to_10_l1090_109081


namespace evaluate_expression_l1090_109006

theorem evaluate_expression (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 17 * a + 7) * (3 * a - 4) = 0 := by
sorry

end evaluate_expression_l1090_109006


namespace units_digit_of_7_pow_6_pow_5_l1090_109062

theorem units_digit_of_7_pow_6_pow_5 : 7^(6^5) % 10 = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l1090_109062


namespace sqrt_neg_two_squared_l1090_109019

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end sqrt_neg_two_squared_l1090_109019


namespace solve_simple_interest_l1090_109056

def simple_interest_problem (rate : ℚ) (principal : ℚ) (interest_difference : ℚ) : Prop :=
  let simple_interest := principal - interest_difference
  let years := simple_interest / (principal * rate)
  years = 5

theorem solve_simple_interest :
  simple_interest_problem (4/100) 3000 2400 := by
  sorry

end solve_simple_interest_l1090_109056


namespace labor_tools_problem_l1090_109020

/-- The price per tool of the first batch of type A labor tools -/
def first_batch_price (total_cost : ℕ) (quantity : ℕ) : ℕ :=
  total_cost / quantity

/-- The price per tool of the second batch of type A labor tools -/
def second_batch_price (first_price : ℕ) (increase : ℕ) : ℕ :=
  first_price + increase

/-- The total cost of the second batch -/
def second_batch_total_cost (price : ℕ) (quantity : ℕ) : ℕ :=
  price * quantity

/-- The maximum number of type A tools in the third batch -/
def max_type_a_tools (type_a_price : ℕ) (type_b_price : ℕ) (total_tools : ℕ) (max_cost : ℕ) : ℕ :=
  min (total_tools) ((max_cost - type_b_price * total_tools) / (type_a_price - type_b_price))

theorem labor_tools_problem (first_total_cost second_total_cost : ℕ) 
  (price_increase : ℕ) (third_batch_total : ℕ) (type_b_price : ℕ) (third_batch_max_cost : ℕ) :
  first_total_cost = 2000 ∧ 
  second_total_cost = 2200 ∧ 
  price_increase = 5 ∧
  third_batch_total = 50 ∧
  type_b_price = 40 ∧
  third_batch_max_cost = 2500 →
  ∃ quantity : ℕ,
    first_batch_price first_total_cost quantity = 50 ∧
    second_batch_price (first_batch_price first_total_cost quantity) price_increase = 
      first_batch_price second_total_cost quantity ∧
    max_type_a_tools 
      (second_batch_price (first_batch_price first_total_cost quantity) price_increase)
      type_b_price third_batch_total third_batch_max_cost = 33 := by
  sorry

end labor_tools_problem_l1090_109020


namespace increase_by_ten_six_times_l1090_109039

/-- Given a starting number of 540, increased by 10 for 6 times, the result is 600 -/
theorem increase_by_ten_six_times : 
  let start : ℕ := 540
  let increase : ℕ := 10
  let times : ℕ := 6
  start + increase * times = 600 := by sorry

end increase_by_ten_six_times_l1090_109039


namespace vanessa_phone_pictures_l1090_109027

theorem vanessa_phone_pictures :
  ∀ (phone_pics camera_pics num_albums pics_per_album : ℕ),
    camera_pics = 7 →
    num_albums = 5 →
    pics_per_album = 6 →
    phone_pics + camera_pics = num_albums * pics_per_album →
    phone_pics = 23 := by
  sorry

end vanessa_phone_pictures_l1090_109027


namespace sum_of_three_smallest_solutions_l1090_109017

theorem sum_of_three_smallest_solutions : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (∀ x : ℝ, x > 0 → x - ⌊x⌋ = 1 / ⌊x⌋ + 1 / ⌊x⌋^2 → x ≥ x₁) ∧
    (∀ x : ℝ, x > 0 → x - ⌊x⌋ = 1 / ⌊x⌋ + 1 / ⌊x⌋^2 → x = x₁ ∨ x ≥ x₂) ∧
    (∀ x : ℝ, x > 0 → x - ⌊x⌋ = 1 / ⌊x⌋ + 1 / ⌊x⌋^2 → x = x₁ ∨ x = x₂ ∨ x ≥ x₃) ∧
    x₁ - ⌊x₁⌋ = 1 / ⌊x₁⌋ + 1 / ⌊x₁⌋^2 ∧
    x₂ - ⌊x₂⌋ = 1 / ⌊x₂⌋ + 1 / ⌊x₂⌋^2 ∧
    x₃ - ⌊x₃⌋ = 1 / ⌊x₃⌋ + 1 / ⌊x₃⌋^2 ∧
    x₁ + x₂ + x₃ = 21/2 := by
  sorry

end sum_of_three_smallest_solutions_l1090_109017


namespace football_practice_hours_l1090_109002

/-- Calculates the total practice hours for a football team in a week with one missed day -/
theorem football_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 6 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 36 := by
sorry

end football_practice_hours_l1090_109002


namespace different_course_selections_l1090_109018

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem different_course_selections (total_courses : ℕ) (courses_per_person : ℕ) : 
  total_courses = 4 → courses_per_person = 2 →
  (choose total_courses courses_per_person * choose total_courses courses_per_person) - 
  (choose total_courses courses_per_person) = 30 := by
  sorry

end different_course_selections_l1090_109018


namespace conor_weekly_vegetables_l1090_109044

/-- Represents the number of vegetables Conor can chop in a day -/
structure DailyVegetables where
  eggplants : ℕ
  carrots : ℕ
  potatoes : ℕ
  onions : ℕ
  zucchinis : ℕ

/-- Calculates the total number of vegetables chopped in a day -/
def DailyVegetables.total (d : DailyVegetables) : ℕ :=
  d.eggplants + d.carrots + d.potatoes + d.onions + d.zucchinis

/-- Conor's chopping rate from Monday to Wednesday -/
def earlyWeekRate : DailyVegetables :=
  { eggplants := 12
    carrots := 9
    potatoes := 8
    onions := 15
    zucchinis := 7 }

/-- Conor's chopping rate from Thursday to Saturday -/
def lateWeekRate : DailyVegetables :=
  { eggplants := 7
    carrots := 5
    potatoes := 4
    onions := 10
    zucchinis := 4 }

/-- The number of days Conor works in the early part of the week -/
def earlyWeekDays : ℕ := 3

/-- The number of days Conor works in the late part of the week -/
def lateWeekDays : ℕ := 3

/-- Theorem: Conor can chop 243 vegetables in a week -/
theorem conor_weekly_vegetables : 
  earlyWeekDays * earlyWeekRate.total + lateWeekDays * lateWeekRate.total = 243 := by
  sorry

end conor_weekly_vegetables_l1090_109044


namespace perfect_square_expression_l1090_109095

theorem perfect_square_expression : ∃ x : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = x ^ 2 := by
  sorry

end perfect_square_expression_l1090_109095


namespace sequence_formula_l1090_109032

theorem sequence_formula (a : ℕ → ℚ) :
  a 1 = 1/4 ∧
  (∀ n : ℕ, n ≥ 2 → a n = 1/2 * a (n-1) + 1/(2^n)) →
  ∀ n : ℕ, n ≥ 1 → a n = (2*n - 1) / (2^(n+1)) :=
by sorry

end sequence_formula_l1090_109032


namespace constant_term_expansion_l1090_109090

theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (x^2 + 2) * (1/x^2 - 1)^5
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = 3 :=
by
  sorry

end constant_term_expansion_l1090_109090


namespace probability_of_black_piece_l1090_109099

def total_pieces : ℕ := 15
def black_pieces : ℕ := 10
def white_pieces : ℕ := 5

theorem probability_of_black_piece :
  (black_pieces : ℚ) / total_pieces = 2 / 3 := by sorry

end probability_of_black_piece_l1090_109099


namespace equal_roots_quadratic_l1090_109093

/-- 
A quadratic equation x^2 - 2x + 2a = 0 has two equal real roots if and only if a = 1/2.
-/
theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + 2*a = 0 ∧ (∀ y : ℝ, y^2 - 2*y + 2*a = 0 → y = x)) ↔ a = 1/2 := by
sorry

end equal_roots_quadratic_l1090_109093


namespace a_range_proof_l1090_109042

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem a_range_proof (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_decreasing f (-1) 1)
  (h2 : f (2*a - 1) < f (1 - a)) :
  2/3 < a ∧ a < 1 := by
  sorry

end a_range_proof_l1090_109042


namespace inequality_implication_l1090_109028

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_implication_l1090_109028


namespace eight_packages_l1090_109098

/-- Given a total number of markers and the number of markers per package,
    calculate the number of packages. -/
def calculate_packages (total_markers : ℕ) (markers_per_package : ℕ) : ℕ :=
  total_markers / markers_per_package

/-- Proof that there are 8 packages when there are 40 markers in total
    and 5 markers per package. -/
theorem eight_packages :
  calculate_packages 40 5 = 8 := by
  sorry


end eight_packages_l1090_109098


namespace jerry_cereal_calories_l1090_109074

/-- Represents the calorie content of Jerry's breakfast items -/
structure BreakfastCalories where
  pancakeCount : ℕ
  pancakeCalories : ℕ
  baconCount : ℕ
  baconCalories : ℕ
  totalCalories : ℕ

/-- Calculates the calories in the cereal bowl given the breakfast composition -/
def cerealCalories (b : BreakfastCalories) : ℕ :=
  b.totalCalories - (b.pancakeCount * b.pancakeCalories + b.baconCount * b.baconCalories)

/-- Theorem stating that Jerry's cereal bowl contains 200 calories -/
theorem jerry_cereal_calories :
  let jerryBreakfast : BreakfastCalories := {
    pancakeCount := 6,
    pancakeCalories := 120,
    baconCount := 2,
    baconCalories := 100,
    totalCalories := 1120
  }
  cerealCalories jerryBreakfast = 200 := by
  sorry

end jerry_cereal_calories_l1090_109074


namespace inequality_proof_l1090_109035

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end inequality_proof_l1090_109035


namespace transmission_time_approx_five_minutes_l1090_109052

/-- Represents the data transmission problem --/
structure DataTransmission where
  numBlocks : ℕ
  chunksPerBlock : ℕ
  errorCorrectionRate : ℚ
  transmissionRate : ℕ

/-- Calculates the total number of chunks including error correction --/
def totalChunks (d : DataTransmission) : ℚ :=
  (d.numBlocks * d.chunksPerBlock : ℚ) * (1 + d.errorCorrectionRate)

/-- Calculates the transmission time in seconds --/
def transmissionTimeSeconds (d : DataTransmission) : ℚ :=
  totalChunks d / d.transmissionRate

/-- Calculates the transmission time in minutes --/
def transmissionTimeMinutes (d : DataTransmission) : ℚ :=
  transmissionTimeSeconds d / 60

/-- The main theorem stating that the transmission time is approximately 5 minutes --/
theorem transmission_time_approx_five_minutes
  (d : DataTransmission)
  (h1 : d.numBlocks = 50)
  (h2 : d.chunksPerBlock = 500)
  (h3 : d.errorCorrectionRate = 1/10)
  (h4 : d.transmissionRate = 100) :
  ∃ ε > 0, |transmissionTimeMinutes d - 5| < ε :=
sorry


end transmission_time_approx_five_minutes_l1090_109052


namespace point_outside_circle_l1090_109065

/-- Given a circle defined by x^2 + y^2 - 2ax + a^2 - a = 0, if the point (a, a) can be outside this circle, then a > 1. -/
theorem point_outside_circle (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + a^2 - a = 0) → 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + a^2 - a < 0 ∧ x = a ∧ y = a) → 
  a > 1 :=
by sorry

end point_outside_circle_l1090_109065


namespace quadratic_inequality_coefficient_sum_l1090_109057

/-- Given a quadratic inequality x^2 + bx - a < 0 with solution set {x | 3 < x < 4},
    prove that the sum of coefficients a + b = -19 -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) :
  (∀ x : ℝ, x^2 + b*x - a < 0 ↔ 3 < x ∧ x < 4) →
  a + b = -19 := by
  sorry

end quadratic_inequality_coefficient_sum_l1090_109057


namespace estate_area_calculation_l1090_109034

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 300

/-- Represents the length of the estate on the map in inches -/
def map_length : ℝ := 6

/-- Represents the width of the estate on the map in inches -/
def map_width : ℝ := 4

/-- Calculates the actual length of the estate in miles -/
def actual_length : ℝ := scale * map_length

/-- Calculates the actual width of the estate in miles -/
def actual_width : ℝ := scale * map_width

/-- Calculates the area of the estate in square miles -/
def estate_area : ℝ := actual_length * actual_width

theorem estate_area_calculation : estate_area = 2160000 := by
  sorry

end estate_area_calculation_l1090_109034


namespace sweets_distribution_l1090_109091

theorem sweets_distribution (total_sweets : ℕ) (sweets_per_child : ℕ) : 
  total_sweets = 288 →
  sweets_per_child = 4 →
  ∃ (num_children : ℕ), 
    (num_children * sweets_per_child + total_sweets / 3 = total_sweets) ∧
    num_children = 48 := by
  sorry

end sweets_distribution_l1090_109091


namespace sqrt_three_irrational_l1090_109068

theorem sqrt_three_irrational :
  (∃ (a b : ℤ), -2 = a / b) →
  (∃ (c d : ℤ), 0 = c / d) →
  (∃ (e f : ℤ), -1/2 = e / f) →
  ¬(∃ (x y : ℤ), Real.sqrt 3 = x / y) :=
by sorry

end sqrt_three_irrational_l1090_109068


namespace ratio_cube_square_l1090_109075

theorem ratio_cube_square (x y : ℝ) (h : x / y = 7 / 5) : 
  x^3 / y^2 = 343 / 25 := by
  sorry

end ratio_cube_square_l1090_109075


namespace cars_in_first_section_l1090_109077

/-- The number of rows in the first section -/
def first_section_rows : ℕ := 15

/-- The number of cars per row in the first section -/
def first_section_cars_per_row : ℕ := 10

/-- The number of rows in the second section -/
def second_section_rows : ℕ := 20

/-- The number of cars per row in the second section -/
def second_section_cars_per_row : ℕ := 9

/-- The number of cars Nate can walk past per minute -/
def cars_passed_per_minute : ℕ := 11

/-- The number of minutes Nate spent searching -/
def search_time_minutes : ℕ := 30

/-- The theorem stating the number of cars in the first section -/
theorem cars_in_first_section :
  first_section_rows * first_section_cars_per_row = 150 := by
  sorry

end cars_in_first_section_l1090_109077


namespace average_problem_l1090_109086

theorem average_problem (x : ℝ) : (20 + 30 + 40 + x) / 4 = 35 → x = 50 := by
  sorry

end average_problem_l1090_109086


namespace mirror_area_l1090_109040

/-- Given a rectangular frame with outer dimensions 70 cm by 100 cm and a uniform frame width of 15 cm,
    the area of the mirror that fits exactly inside the frame is 2800 cm². -/
theorem mirror_area (frame_outer_width : ℕ) (frame_outer_height : ℕ) (frame_thickness : ℕ) :
  frame_outer_width = 100 ∧ frame_outer_height = 70 ∧ frame_thickness = 15 →
  (frame_outer_width - 2 * frame_thickness) * (frame_outer_height - 2 * frame_thickness) = 2800 :=
by sorry

end mirror_area_l1090_109040


namespace quadratic_roots_relation_l1090_109076

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = m ∧
   (3 * r₁) + (3 * r₂) = -m ∧ (3 * r₁) * (3 * r₂) = n) →
  n / p = -9 := by
sorry

end quadratic_roots_relation_l1090_109076


namespace min_value_of_expression_l1090_109085

theorem min_value_of_expression (n : ℕ) (h : 10 ≤ n ∧ n ≤ 99) : 
  3 * (300 - n) ≥ 603 :=
by sorry

end min_value_of_expression_l1090_109085


namespace triangle_abc_properties_l1090_109092

theorem triangle_abc_properties (a b c A B C : ℝ) : 
  b = Real.sqrt 2 →
  c = 1 →
  Real.cos B = 3/4 →
  (Real.sin C = Real.sqrt 14 / 8 ∧ 
   Real.sin A * b * c / 2 = Real.sqrt 7 / 4) := by
  sorry

end triangle_abc_properties_l1090_109092


namespace ant_walk_probability_l1090_109004

/-- Represents a point on the lattice -/
structure Point where
  x : Int
  y : Int

/-- Determines if a point is red (even x+y) or blue (odd x+y) -/
def isRed (p : Point) : Bool :=
  (p.x + p.y) % 2 == 0

/-- Represents the ant's position and the number of steps taken -/
structure AntState where
  position : Point
  steps : Nat

/-- Defines the possible directions the ant can move -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Returns the new point after moving in a given direction -/
def move (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.Up => ⟨p.x, p.y + 1⟩
  | Direction.Down => ⟨p.x, p.y - 1⟩
  | Direction.Left => ⟨p.x - 1, p.y⟩
  | Direction.Right => ⟨p.x + 1, p.y⟩

/-- Defines the probability of the ant being at point C after 4 steps -/
def probAtCAfter4Steps (startPoint : Point) (endPoint : Point) : Real :=
  sorry

/-- The main theorem to prove -/
theorem ant_walk_probability :
  probAtCAfter4Steps ⟨0, 0⟩ ⟨1, 0⟩ = 1/3 := by
  sorry

end ant_walk_probability_l1090_109004


namespace bucket_capacity_is_seven_l1090_109059

/-- The capacity of the tank in litres -/
def tank_capacity : ℕ := 12 * 49

/-- The number of buckets needed in the second scenario -/
def buckets_second_scenario : ℕ := 84

/-- The capacity of each bucket in the second scenario -/
def bucket_capacity_second_scenario : ℚ := tank_capacity / buckets_second_scenario

theorem bucket_capacity_is_seven :
  bucket_capacity_second_scenario = 7 := by
  sorry

end bucket_capacity_is_seven_l1090_109059


namespace system_of_equations_solutions_l1090_109096

theorem system_of_equations_solutions :
  -- System 1
  (∃ (x y : ℚ), x = 1 - y ∧ 3 * x + y = 1 ∧ x = 0 ∧ y = 1) ∧
  -- System 2
  (∃ (x y : ℚ), 3 * x + y = 18 ∧ 2 * x - y = -11 ∧ x = 7/5 ∧ y = 69/5) :=
by sorry

end system_of_equations_solutions_l1090_109096


namespace geometric_sequence_properties_l1090_109080

/-- A geometric sequence with given second and fifth terms -/
structure GeometricSequence where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- The third term of the geometric sequence -/
def third_term (g : GeometricSequence) : ℝ := 49

/-- The sum of the first four terms of the geometric sequence -/
def sum_first_four (g : GeometricSequence) : ℝ := 183.75

/-- Theorem stating the properties of the geometric sequence -/
theorem geometric_sequence_properties (g : GeometricSequence) :
  third_term g = 49 ∧ sum_first_four g = 183.75 := by
  sorry


end geometric_sequence_properties_l1090_109080


namespace road_trip_total_hours_l1090_109063

/-- Calculates the total hours spent on a road trip -/
def total_road_trip_hours (jade_hours : Fin 3 → ℕ) (krista_hours : Fin 3 → ℕ) (break_hours : ℕ) : ℕ :=
  (Finset.sum Finset.univ (λ i => jade_hours i + krista_hours i)) + 3 * break_hours

theorem road_trip_total_hours : 
  let jade_hours : Fin 3 → ℕ := ![8, 7, 6]
  let krista_hours : Fin 3 → ℕ := ![6, 5, 4]
  let break_hours : ℕ := 2
  total_road_trip_hours jade_hours krista_hours break_hours = 42 := by
  sorry

#eval total_road_trip_hours ![8, 7, 6] ![6, 5, 4] 2

end road_trip_total_hours_l1090_109063


namespace morning_routine_time_l1090_109025

def skincare_routine_time : ℕ := 2 + 3 + 3 + 4 + 1 + 3 + 2 + 5 + 2 + 2 + 1

def makeup_time : ℕ := 30

def hair_styling_time : ℕ := 20

theorem morning_routine_time :
  skincare_routine_time + makeup_time + hair_styling_time = 78 := by
  sorry

end morning_routine_time_l1090_109025
