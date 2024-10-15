import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_condition_l190_19084

theorem perfect_square_condition (Z K : ℤ) : 
  (1000 < Z) → (Z < 5000) → (K > 1) → (Z = K * K^2) → 
  (∃ (n : ℤ), Z = n^2) → (K = 16) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l190_19084


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_one_range_of_m_for_nonempty_solution_l190_19093

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x - 2| + |x + m|

-- Theorem for part (1)
theorem solution_set_when_m_is_one :
  ∃ (a b : ℝ), a = 0 ∧ b = 4/3 ∧
  (∀ x, f x 1 ≤ 3 ↔ a ≤ x ∧ x ≤ b) :=
sorry

-- Theorem for part (2)
theorem range_of_m_for_nonempty_solution :
  ∃ (lower upper : ℝ), lower = -4 ∧ upper = 2 ∧
  (∀ m, (∃ x, f x m ≤ 3) ↔ lower ≤ m ∧ m ≤ upper) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_one_range_of_m_for_nonempty_solution_l190_19093


namespace NUMINAMATH_CALUDE_functional_equation_solution_l190_19062

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l190_19062


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l190_19016

/-- The volume of a wedge from a sphere -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) : 
  circumference = 18 * Real.pi → num_wedges = 6 →
  (4 / 3 * Real.pi * (circumference / (2 * Real.pi))^3) / num_wedges = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l190_19016


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l190_19096

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := 2 * n - 12

-- Define the geometric sequence b_n
def b (n : ℕ) : ℝ := -8

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := -8 * n

theorem arithmetic_and_geometric_sequences :
  (a 3 = -6) ∧ 
  (a 6 = 0) ∧ 
  (b 1 = -8) ∧ 
  (b 2 = a 1 + a 2 + a 3) ∧
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧  -- arithmetic sequence property
  (∀ n : ℕ, b (n + 1) / b n = b 2 / b 1) ∧  -- geometric sequence property
  (∀ n : ℕ, S n = (1 - (b 2 / b 1)^n) / (1 - b 2 / b 1) * b 1) -- sum formula for geometric sequence
  :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l190_19096


namespace NUMINAMATH_CALUDE_definite_integral_sqrt_4_minus_x_squared_minus_2x_l190_19027

theorem definite_integral_sqrt_4_minus_x_squared_minus_2x : 
  ∫ (x : ℝ) in (0)..(2), (Real.sqrt (4 - x^2) - 2*x) = π - 4 := by sorry

end NUMINAMATH_CALUDE_definite_integral_sqrt_4_minus_x_squared_minus_2x_l190_19027


namespace NUMINAMATH_CALUDE_cookie_distribution_l190_19013

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : total_cookies = 35)
  (h2 : num_people = 5)
  (h3 : total_cookies = num_people * cookies_per_person) :
  cookies_per_person = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l190_19013


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l190_19006

theorem quadratic_completion_square (a : ℝ) (n : ℝ) : 
  (∀ x, x^2 + a*x + (1/4 : ℝ) = (x + n)^2 + (1/16 : ℝ)) → 
  a < 0 → 
  a = -((3 : ℝ).sqrt / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l190_19006


namespace NUMINAMATH_CALUDE_power_multiplication_division_equality_l190_19085

theorem power_multiplication_division_equality : (12 : ℚ)^2 * 6^3 / 432 = 72 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_division_equality_l190_19085


namespace NUMINAMATH_CALUDE_keychain_cost_is_five_l190_19024

/-- The cost of a bracelet in dollars -/
def bracelet_cost : ℝ := 4

/-- The cost of a coloring book in dollars -/
def coloring_book_cost : ℝ := 3

/-- The cost of Paula's purchase in dollars -/
def paula_cost (keychain_cost : ℝ) : ℝ := 2 * bracelet_cost + keychain_cost

/-- The cost of Olive's purchase in dollars -/
def olive_cost : ℝ := coloring_book_cost + bracelet_cost

/-- The total amount spent by Paula and Olive in dollars -/
def total_spent : ℝ := 20

/-- Theorem stating that the keychain cost is 5 dollars -/
theorem keychain_cost_is_five : 
  ∃ (keychain_cost : ℝ), paula_cost keychain_cost + olive_cost = total_spent ∧ keychain_cost = 5 :=
sorry

end NUMINAMATH_CALUDE_keychain_cost_is_five_l190_19024


namespace NUMINAMATH_CALUDE_solution_set_equality_l190_19000

def solution_set : Set ℝ := {x : ℝ | |x - 1| - |x - 5| < 2}

theorem solution_set_equality : solution_set = Set.Iio 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l190_19000


namespace NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l190_19098

theorem sum_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (fun i => 12 * (i + 1)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_smallest_multiples_of_12_l190_19098


namespace NUMINAMATH_CALUDE_photo_album_distribution_l190_19077

/-- Represents the distribution of photos in an album --/
structure PhotoAlbum where
  total_photos : ℕ
  total_pages : ℕ
  photos_per_page_set1 : ℕ
  photos_per_page_set2 : ℕ
  photos_per_page_remaining : ℕ

/-- Theorem stating the correct distribution of pages for the given photo album --/
theorem photo_album_distribution (album : PhotoAlbum) 
  (h1 : album.total_photos = 100)
  (h2 : album.total_pages = 30)
  (h3 : album.photos_per_page_set1 = 3)
  (h4 : album.photos_per_page_set2 = 4)
  (h5 : album.photos_per_page_remaining = 3) :
  ∃ (pages_set1 pages_set2 pages_remaining : ℕ),
    pages_set1 = 0 ∧ 
    pages_set2 = 10 ∧
    pages_remaining = 20 ∧
    pages_set1 + pages_set2 + pages_remaining = album.total_pages ∧
    album.photos_per_page_set1 * pages_set1 + 
    album.photos_per_page_set2 * pages_set2 + 
    album.photos_per_page_remaining * pages_remaining = album.total_photos :=
by
  sorry

end NUMINAMATH_CALUDE_photo_album_distribution_l190_19077


namespace NUMINAMATH_CALUDE_matt_received_more_than_lauren_l190_19094

-- Define the given conditions
def total_pencils : ℕ := 2 * 12
def pencils_to_lauren : ℕ := 6
def pencils_left : ℕ := 9

-- Define the number of pencils Matt received
def pencils_to_matt : ℕ := total_pencils - pencils_to_lauren - pencils_left

-- Theorem to prove
theorem matt_received_more_than_lauren : 
  pencils_to_matt - pencils_to_lauren = 3 := by
sorry

end NUMINAMATH_CALUDE_matt_received_more_than_lauren_l190_19094


namespace NUMINAMATH_CALUDE_nested_fraction_equals_nineteen_elevenths_l190_19043

theorem nested_fraction_equals_nineteen_elevenths :
  1 + 1 / (1 + 1 / (2 + 2 / 3)) = 19 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_nineteen_elevenths_l190_19043


namespace NUMINAMATH_CALUDE_sin_330_degrees_l190_19059

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l190_19059


namespace NUMINAMATH_CALUDE_modulus_of_one_minus_i_l190_19031

theorem modulus_of_one_minus_i :
  let z : ℂ := 1 - Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_minus_i_l190_19031


namespace NUMINAMATH_CALUDE_ship_speed_calculation_l190_19037

theorem ship_speed_calculation (total_distance : ℝ) (travel_time : ℝ) (backward_distance : ℝ) :
  travel_time = 20 ∧
  backward_distance = 200 ∧
  total_distance / 2 - total_distance / 3 = backward_distance →
  (total_distance / 2) / travel_time = 30 := by
sorry

end NUMINAMATH_CALUDE_ship_speed_calculation_l190_19037


namespace NUMINAMATH_CALUDE_parallelogram_problem_l190_19079

-- Define a parallelogram
structure Parallelogram :=
  (EF GH FG HE : ℝ)
  (is_parallelogram : EF = GH ∧ FG = HE)

-- Define the problem
theorem parallelogram_problem (EFGH : Parallelogram)
  (h1 : EFGH.EF = 52)
  (h2 : ∃ z : ℝ, EFGH.FG = 2 * z^4)
  (h3 : ∃ w : ℝ, EFGH.GH = 3 * w + 6)
  (h4 : EFGH.HE = 16) :
  ∃ w z : ℝ, w * z = 46 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_problem_l190_19079


namespace NUMINAMATH_CALUDE_no_reversed_arithmetic_progression_l190_19036

/-- Function that returns the odd positive integer obtained by reversing the binary representation of n -/
def r (n : Nat) : Nat :=
  sorry

/-- Predicate to check if a sequence is an arithmetic progression -/
def isArithmeticProgression (s : List Nat) : Prop :=
  sorry

theorem no_reversed_arithmetic_progression :
  ¬∃ (a : Fin 8 → Nat),
    (∀ i : Fin 8, Odd (a i)) ∧
    (∀ i j : Fin 8, i < j → a i < a j) ∧
    isArithmeticProgression (List.ofFn a) ∧
    isArithmeticProgression (List.map r (List.ofFn a)) :=
  sorry

end NUMINAMATH_CALUDE_no_reversed_arithmetic_progression_l190_19036


namespace NUMINAMATH_CALUDE_marla_nightly_cost_l190_19067

/-- Represents the exchange rates and Marla's scavenging situation in the post-apocalyptic wasteland -/
structure WastelandEconomy where
  lizard_to_caps : ℕ → ℕ
  lizards_to_water : ℕ → ℕ
  horse_to_water : ℕ
  daily_scavenge : ℕ
  days_to_horse : ℕ

/-- Calculates the number of bottle caps Marla needs to pay per night for food and shelter -/
def nightly_cost (we : WastelandEconomy) : ℕ :=
  -- The actual calculation goes here, but we'll use sorry to skip the proof
  sorry

/-- Theorem stating that in the given wasteland economy, Marla needs to pay 4 bottle caps per night -/
theorem marla_nightly_cost :
  let we : WastelandEconomy := {
    lizard_to_caps := λ n => 8 * n,
    lizards_to_water := λ n => (5 * n) / 3,
    horse_to_water := 80,
    daily_scavenge := 20,
    days_to_horse := 24
  }
  nightly_cost we = 4 := by
  sorry

end NUMINAMATH_CALUDE_marla_nightly_cost_l190_19067


namespace NUMINAMATH_CALUDE_johnny_red_pencils_l190_19009

/-- The number of red pencils Johnny bought -/
def total_red_pencils (total_packs : ℕ) (regular_red_per_pack : ℕ) 
  (extra_red_packs_1 : ℕ) (extra_red_per_pack_1 : ℕ)
  (extra_red_packs_2 : ℕ) (extra_red_per_pack_2 : ℕ) : ℕ :=
  total_packs * regular_red_per_pack + 
  extra_red_packs_1 * extra_red_per_pack_1 +
  extra_red_packs_2 * extra_red_per_pack_2

/-- Theorem: Johnny bought 46 red pencils -/
theorem johnny_red_pencils : 
  total_red_pencils 25 1 5 3 6 1 = 46 := by
  sorry

end NUMINAMATH_CALUDE_johnny_red_pencils_l190_19009


namespace NUMINAMATH_CALUDE_least_positive_integer_l190_19052

theorem least_positive_integer (x : ℕ) : x = 6 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → ¬((2*y)^2 + 2*41*(2*y) + 41^2) % 53 = 0) ∧
  ((2*x)^2 + 2*41*(2*x) + 41^2) % 53 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_l190_19052


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l190_19054

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def r : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n r

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l190_19054


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l190_19007

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle structure -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem statement -/
theorem parabola_line_intersection (C : Parabola) (l : Line) (M N : Point) (directrix : Line) :
  (l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3) →  -- Line equation: y = -√3(x-1)
  (Point.mk (C.p / 2) 0).y = l.m * (Point.mk (C.p / 2) 0).x + l.b →  -- Line passes through focus
  (M.y^2 = 2 * C.p * M.x ∧ N.y^2 = 2 * C.p * N.x) →  -- M and N are on the parabola
  (directrix.m = 0 ∧ directrix.b = -C.p / 2) →  -- Directrix equation: x = -p/2
  (C.p = 2 ∧  -- First conclusion: p = 2
   ∃ (circ : Circle), circ.center = Point.mk ((M.x + N.x) / 2) ((M.y + N.y) / 2) ∧
                      circ.radius = abs ((M.x - N.x) / 2) ∧
                      abs (circ.center.x - (-C.p / 2)) = circ.radius)  -- Second conclusion: Circle tangent to directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l190_19007


namespace NUMINAMATH_CALUDE_mixture_salt_concentration_l190_19090

/-- Represents the concentration of a solution as a real number between 0 and 1 -/
def Concentration := { c : ℝ // 0 ≤ c ∧ c ≤ 1 }

/-- Calculates the concentration of salt in a mixture of pure water and salt solution -/
def mixtureSaltConcentration (pureWaterVolume : ℝ) (saltSolutionVolume : ℝ) (saltSolutionConcentration : Concentration) : Concentration :=
  sorry

/-- Theorem: The concentration of salt in a mixture of 1 liter of pure water and 0.2 liters of 60% salt solution is 10% -/
theorem mixture_salt_concentration :
  let pureWaterVolume : ℝ := 1
  let saltSolutionVolume : ℝ := 0.2
  let saltSolutionConcentration : Concentration := ⟨0.6, by sorry⟩
  let resultingConcentration : Concentration := mixtureSaltConcentration pureWaterVolume saltSolutionVolume saltSolutionConcentration
  resultingConcentration.val = 0.1 := by sorry

end NUMINAMATH_CALUDE_mixture_salt_concentration_l190_19090


namespace NUMINAMATH_CALUDE_line_intersection_with_x_axis_l190_19025

/-- A line passing through two given points intersects the x-axis at a specific point. -/
theorem line_intersection_with_x_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 2 ∧ y₁ = 3) 
  (h_point2 : x₂ = -4 ∧ y₂ = 9) : 
  ∃ x : ℝ, x = 5 ∧ (y₂ - y₁) * x + (x₁ * y₂ - x₂ * y₁) = (y₂ - y₁) * x₁ := by
  sorry

#check line_intersection_with_x_axis

end NUMINAMATH_CALUDE_line_intersection_with_x_axis_l190_19025


namespace NUMINAMATH_CALUDE_investment_return_percentage_l190_19065

/-- Proves that the yearly return percentage of a $500 investment is 7% given specific conditions --/
theorem investment_return_percentage : 
  ∀ (total_investment small_investment large_investment : ℝ)
    (combined_return_rate small_return_rate large_return_rate : ℝ),
  total_investment = 2000 →
  small_investment = 500 →
  large_investment = 1500 →
  combined_return_rate = 0.10 →
  large_return_rate = 0.11 →
  combined_return_rate * total_investment = 
    small_return_rate * small_investment + large_return_rate * large_investment →
  small_return_rate = 0.07 := by
sorry


end NUMINAMATH_CALUDE_investment_return_percentage_l190_19065


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l190_19051

theorem triangle_angle_difference (a b c : ℝ) : 
  a = 32 →
  b = 96 →
  c = 52 →
  b = 3 * a →
  2 * a - c = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l190_19051


namespace NUMINAMATH_CALUDE_max_of_roots_l190_19072

theorem max_of_roots (α β γ : ℝ) 
  (sum_eq : α + β + γ = 14)
  (sum_squares_eq : α^2 + β^2 + γ^2 = 84)
  (sum_cubes_eq : α^3 + β^3 + γ^3 = 584) :
  max α (max β γ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_of_roots_l190_19072


namespace NUMINAMATH_CALUDE_clarence_oranges_l190_19005

/-- The number of oranges Clarence has initially -/
def initial_oranges : ℕ := 5

/-- The number of oranges Clarence receives from Joyce -/
def oranges_from_joyce : ℕ := 3

/-- The total number of oranges Clarence has -/
def total_oranges : ℕ := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_l190_19005


namespace NUMINAMATH_CALUDE_total_subscription_is_50000_l190_19022

/-- Represents the subscription amounts and profit distribution for a business --/
structure BusinessSubscription where
  /-- C's subscription amount --/
  c : ℕ
  /-- Total profit --/
  total_profit : ℕ
  /-- A's profit share --/
  a_profit : ℕ

/-- Calculates the total subscription amount given the business subscription details --/
def total_subscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c + 14000

/-- Theorem stating that the total subscription amount is 50000 given the problem conditions --/
theorem total_subscription_is_50000 (bs : BusinessSubscription)
  (h1 : bs.total_profit = 36000)
  (h2 : bs.a_profit = 15120)
  (h3 : bs.a_profit * (3 * bs.c + 14000) = bs.total_profit * (bs.c + 9000)) :
  total_subscription bs = 50000 := by
  sorry

#check total_subscription_is_50000

end NUMINAMATH_CALUDE_total_subscription_is_50000_l190_19022


namespace NUMINAMATH_CALUDE_first_day_sale_is_30_percent_l190_19026

/-- The percentage of apples sold on the first day -/
def first_day_sale_percentage : ℝ := sorry

/-- The percentage of apples thrown away on the first day -/
def first_day_throwaway_percentage : ℝ := 0.20

/-- The percentage of apples sold on the second day -/
def second_day_sale_percentage : ℝ := 0.50

/-- The total percentage of apples thrown away -/
def total_throwaway_percentage : ℝ := 0.42

/-- Theorem stating that the percentage of apples sold on the first day is 30% -/
theorem first_day_sale_is_30_percent :
  first_day_sale_percentage = 0.30 :=
by
  sorry

end NUMINAMATH_CALUDE_first_day_sale_is_30_percent_l190_19026


namespace NUMINAMATH_CALUDE_debate_team_group_size_l190_19048

theorem debate_team_group_size :
  ∀ (boys girls groups : ℕ),
    boys = 26 →
    girls = 46 →
    groups = 8 →
    (boys + girls) / groups = 9 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l190_19048


namespace NUMINAMATH_CALUDE_sin_transformation_l190_19004

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (x / 3 + π / 6) = 2 * Real.sin ((3 * x + π) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l190_19004


namespace NUMINAMATH_CALUDE_half_percent_as_repeating_decimal_l190_19064

theorem half_percent_as_repeating_decimal : 
  (1 / 2 : ℚ) / 100 = 0.00500 := by sorry

end NUMINAMATH_CALUDE_half_percent_as_repeating_decimal_l190_19064


namespace NUMINAMATH_CALUDE_sock_pairs_theorem_l190_19041

/-- Given an initial number of sock pairs and a number of lost individual socks,
    calculates the maximum number of complete pairs remaining. -/
def maxRemainingPairs (initialPairs : ℕ) (lostSocks : ℕ) : ℕ :=
  initialPairs - min initialPairs lostSocks

/-- Theorem stating that with 25 initial pairs and 12 lost socks,
    the maximum number of complete pairs remaining is 13. -/
theorem sock_pairs_theorem :
  maxRemainingPairs 25 12 = 13 := by
  sorry

#eval maxRemainingPairs 25 12

end NUMINAMATH_CALUDE_sock_pairs_theorem_l190_19041


namespace NUMINAMATH_CALUDE_sequence_property_main_theorem_l190_19058

def sequence_a (n : ℕ+) : ℝ :=
  sorry

theorem sequence_property (n : ℕ+) :
  (Finset.range n).sum (λ i => sequence_a ⟨i + 1, Nat.succ_pos i⟩) = n - sequence_a n :=
sorry

def sequence_b (n : ℕ+) : ℝ :=
  (2 - n) * (sequence_a n - 1)

theorem main_theorem :
  (∃ r : ℝ, ∀ n : ℕ+, sequence_a (n + 1) - 1 = r * (sequence_a n - 1)) ∧
  (∀ t : ℝ, (∀ n : ℕ+, sequence_b n + (1/4) * t ≤ t^2) ↔ t ≤ -1/4 ∨ t ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_main_theorem_l190_19058


namespace NUMINAMATH_CALUDE_line_through_point_l190_19008

/-- Given a line with equation y = 2x + b passing through the point (-4, 0), prove that b = 8 -/
theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → -- The line has equation y = 2x + b
  (0 = 2 * (-4) + b) →         -- The line passes through the point (-4, 0)
  b = 8 :=                     -- The value of b is 8
by sorry

end NUMINAMATH_CALUDE_line_through_point_l190_19008


namespace NUMINAMATH_CALUDE_belle_collected_97_stickers_l190_19076

def belle_stickers (carolyn_stickers : ℕ) (difference : ℕ) : ℕ :=
  carolyn_stickers + difference

theorem belle_collected_97_stickers 
  (h1 : belle_stickers 79 18 = 97) : belle_stickers 79 18 = 97 := by
  sorry

end NUMINAMATH_CALUDE_belle_collected_97_stickers_l190_19076


namespace NUMINAMATH_CALUDE_sibling_age_sum_l190_19035

/-- Given the ages of four siblings with specific relationships, prove that the sum of three of their ages is 25. -/
theorem sibling_age_sum : 
  ∀ (juliet maggie ralph nicky : ℕ),
  juliet = maggie + 3 →
  ralph = juliet + 2 →
  2 * nicky = ralph →
  juliet = 10 →
  maggie + ralph + nicky = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l190_19035


namespace NUMINAMATH_CALUDE_total_equivalent_pencils_is_139_9_l190_19029

/-- Calculates the total equivalent number of pencils in three drawers after additions and removals --/
def totalEquivalentPencils (
  initialPencils1 : Float
  ) (initialPencils2 : Float
  ) (initialPens3 : Float
  ) (mikeAddedPencils1 : Float
  ) (sarahAddedPencils2 : Float
  ) (sarahAddedPens2 : Float
  ) (joeRemovedPencils1 : Float
  ) (joeRemovedPencils2 : Float
  ) (joeRemovedPens3 : Float
  ) (exchangeRate : Float
  ) : Float :=
  let finalPencils1 := initialPencils1 + mikeAddedPencils1 - joeRemovedPencils1
  let finalPencils2 := initialPencils2 + sarahAddedPencils2 - joeRemovedPencils2
  let finalPens3 := initialPens3 + sarahAddedPens2 - joeRemovedPens3
  let totalPencils := finalPencils1 + finalPencils2 + (finalPens3 * exchangeRate)
  totalPencils

theorem total_equivalent_pencils_is_139_9 :
  totalEquivalentPencils 41.5 25.2 13.6 30.7 18.5 8.4 5.3 7.1 3.8 2 = 139.9 := by
  sorry

end NUMINAMATH_CALUDE_total_equivalent_pencils_is_139_9_l190_19029


namespace NUMINAMATH_CALUDE_cricket_team_size_is_eleven_l190_19097

/-- Represents the number of members in a cricket team satisfying specific age conditions. -/
def cricket_team_size : ℕ :=
  let captain_age : ℕ := 28
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 25
  let n : ℕ := 11  -- The number we want to prove

  have h1 : n * team_average_age = (n - 2) * (team_average_age - 1) + captain_age + wicket_keeper_age :=
    by sorry

  n

theorem cricket_team_size_is_eleven : cricket_team_size = 11 := by
  unfold cricket_team_size
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_is_eleven_l190_19097


namespace NUMINAMATH_CALUDE_solution_values_l190_19015

theorem solution_values (x : ℝ) (hx : x^2 + 4 * (x / (x - 2))^2 = 45) :
  let y := ((x - 2)^2 * (x + 3)) / (2*x - 3)
  y = 2 ∨ y = 16 :=
sorry

end NUMINAMATH_CALUDE_solution_values_l190_19015


namespace NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l190_19060

/-- Given a right triangle divided into a rectangle and two smaller right triangles,
    if the area of one small triangle is n times the area of the rectangle,
    then the ratio of the area of the other small triangle to the rectangle is b/(4na) -/
theorem right_triangle_division_area_ratio
  (a b : ℝ)
  (n : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_n : 0 < n)
  (h_ne : a ≠ b)
  (h_area_ratio : ∃ (small_triangle_area rectangle_area : ℝ),
    small_triangle_area = n * rectangle_area ∧
    rectangle_area = a * b) :
  ∃ (other_small_triangle_area rectangle_area : ℝ),
    other_small_triangle_area / rectangle_area = b / (4 * n * a) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l190_19060


namespace NUMINAMATH_CALUDE_pizza_party_children_count_l190_19089

theorem pizza_party_children_count (total : ℕ) (children : ℕ) (adults : ℕ) : 
  total = 120 →
  children = 2 * adults →
  total = children + adults →
  children = 80 := by
sorry

end NUMINAMATH_CALUDE_pizza_party_children_count_l190_19089


namespace NUMINAMATH_CALUDE_probability_no_twos_l190_19049

def valid_id (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 5000 ∧ ¬(String.contains (toString n) '2')

def count_valid_ids : Nat :=
  (List.range 5000).filter valid_id |>.length

theorem probability_no_twos :
  count_valid_ids = 2916 →
  (count_valid_ids : ℚ) / 5000 = 729 / 1250 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_twos_l190_19049


namespace NUMINAMATH_CALUDE_two_solutions_with_more_sheep_l190_19092

def budget : ℕ := 800
def goat_cost : ℕ := 15
def sheep_cost : ℕ := 16

def is_valid_solution (g h : ℕ) : Prop :=
  goat_cost * g + sheep_cost * h = budget ∧ h > g

theorem two_solutions_with_more_sheep :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (g h : ℕ), (g, h) ∈ s ↔ is_valid_solution g h) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_with_more_sheep_l190_19092


namespace NUMINAMATH_CALUDE_triangle345_circle1_common_points_l190_19069

/-- Represents the number of common points between a triangle and a circle -/
inductive CommonPoints
  | Zero
  | One
  | Two
  | Four

/-- A triangle with side lengths 3, 4, and 5 -/
structure Triangle345 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_eq : side1 = 3
  side2_eq : side2 = 4
  side3_eq : side3 = 5

/-- A circle with radius 1 -/
structure Circle1 where
  radius : ℝ
  radius_eq : radius = 1

/-- The theorem stating the possible numbers of common points -/
theorem triangle345_circle1_common_points (t : Triangle345) (c : Circle1) :
  {cp : CommonPoints | cp = CommonPoints.Zero ∨ cp = CommonPoints.One ∨ 
                       cp = CommonPoints.Two ∨ cp = CommonPoints.Four} = 
  {CommonPoints.Zero, CommonPoints.One, CommonPoints.Two, CommonPoints.Four} :=
sorry

end NUMINAMATH_CALUDE_triangle345_circle1_common_points_l190_19069


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l190_19042

theorem wrong_mark_calculation (total_marks : ℝ) : 
  let n : ℕ := 40
  let correct_mark : ℝ := 63
  let wrong_mark : ℝ := (total_marks - correct_mark + n / 2) / (1 - 1 / n)
  wrong_mark = 43 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l190_19042


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l190_19045

/-- A quadratic equation with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + 2 * m * x + m + 1

/-- Condition for the quadratic equation to have two distinct real roots -/
def has_distinct_real_roots (m : ℝ) : Prop :=
  (2 * m)^2 - 4 * (m - 3) * (m + 1) > 0

/-- Condition for the roots not being opposites of each other -/
def roots_not_opposite (m : ℝ) : Prop := m ≠ 0

/-- The range of m satisfying both conditions -/
def valid_m_range (m : ℝ) : Prop :=
  m > -3/2 ∧ m ≠ 0 ∧ m ≠ 3

theorem quadratic_equation_roots_condition :
  ∀ m : ℝ, has_distinct_real_roots m ∧ roots_not_opposite m ↔ valid_m_range m :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l190_19045


namespace NUMINAMATH_CALUDE_village_x_decrease_rate_l190_19061

def village_x_initial_population : ℕ := 68000
def village_y_initial_population : ℕ := 42000
def village_y_growth_rate : ℕ := 800
def years_until_equal : ℕ := 13

theorem village_x_decrease_rate (village_x_decrease_rate : ℕ) : 
  village_x_initial_population - years_until_equal * village_x_decrease_rate = 
  village_y_initial_population + years_until_equal * village_y_growth_rate → 
  village_x_decrease_rate = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_village_x_decrease_rate_l190_19061


namespace NUMINAMATH_CALUDE_production_exceeds_target_l190_19001

/-- The initial production in 2014 -/
def initial_production : ℕ := 40000

/-- The annual increase rate -/
def increase_rate : ℚ := 1/5

/-- The target production to exceed -/
def target_production : ℕ := 120000

/-- The logarithm of 2 -/
def log_2 : ℚ := 3010/10000

/-- The logarithm of 3 -/
def log_3 : ℚ := 4771/10000

/-- The number of years after 2014 when production exceeds the target -/
def years_to_exceed_target : ℕ := 7

theorem production_exceeds_target :
  years_to_exceed_target = 
    (Nat.ceil (log_3 / (increase_rate * log_2))) :=
by sorry

end NUMINAMATH_CALUDE_production_exceeds_target_l190_19001


namespace NUMINAMATH_CALUDE_michaels_brother_money_l190_19040

/-- Given that Michael has $42 and his brother has $17, Michael gives half his money to his brother,
    and his brother then buys $3 worth of candy, prove that his brother ends up with $35. -/
theorem michaels_brother_money (michael_initial : ℕ) (brother_initial : ℕ) 
    (candy_cost : ℕ) (h1 : michael_initial = 42) (h2 : brother_initial = 17) 
    (h3 : candy_cost = 3) : 
    brother_initial + michael_initial / 2 - candy_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_michaels_brother_money_l190_19040


namespace NUMINAMATH_CALUDE_fermat_primes_totient_divisor_641_l190_19073

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

theorem fermat_primes_totient (k : ℕ) : 
  (phi (sigma (2^k)) = 2^k) ↔ k ∈ ({1, 3, 7, 15, 31} : Set ℕ) := by
  sorry

/-- 641 is a divisor of 2^32 + 1 -/
theorem divisor_641 : ∃ m : ℕ, 2^32 + 1 = 641 * m := by
  sorry

end NUMINAMATH_CALUDE_fermat_primes_totient_divisor_641_l190_19073


namespace NUMINAMATH_CALUDE_arrangement_and_selection_theorem_l190_19071

def girls : ℕ := 3
def boys : ℕ := 4
def total_people : ℕ := girls + boys

def arrangements_no_adjacent_girls : ℕ := (Nat.factorial boys) * (Nat.choose (boys + 1) girls)

def selections_with_at_least_one_girl : ℕ := Nat.choose total_people 3 - Nat.choose boys 3

theorem arrangement_and_selection_theorem :
  (arrangements_no_adjacent_girls = 1440) ∧
  (selections_with_at_least_one_girl = 31) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_and_selection_theorem_l190_19071


namespace NUMINAMATH_CALUDE_vet_donation_calculation_l190_19087

/-- Represents the vet fees for different animals --/
structure VetFees where
  dog : ℝ
  cat : ℝ
  rabbit : ℝ
  parrot : ℝ

/-- Represents the number of adoptions for each animal type --/
structure Adoptions where
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  parrots : ℕ

/-- Calculates the total vet fees with discounts applied --/
def calculateTotalFees (fees : VetFees) (adoptions : Adoptions) (multiAdoptDiscount : ℝ) 
    (dogCatAdoptions : ℕ) (parrotRabbitAdoptions : ℕ) : ℝ := sorry

/-- Calculates the vet's donation based on the total fees --/
def calculateDonation (totalFees : ℝ) (donationRate : ℝ) : ℝ := sorry

theorem vet_donation_calculation (fees : VetFees) (adoptions : Adoptions) 
    (multiAdoptDiscount : ℝ) (dogCatAdoptions : ℕ) (parrotRabbitAdoptions : ℕ) 
    (donationRate : ℝ) :
  fees.dog = 15 ∧ fees.cat = 13 ∧ fees.rabbit = 10 ∧ fees.parrot = 12 ∧
  adoptions.dogs = 8 ∧ adoptions.cats = 3 ∧ adoptions.rabbits = 5 ∧ adoptions.parrots = 2 ∧
  multiAdoptDiscount = 0.1 ∧ dogCatAdoptions = 2 ∧ parrotRabbitAdoptions = 1 ∧
  donationRate = 1/3 →
  calculateDonation (calculateTotalFees fees adoptions multiAdoptDiscount dogCatAdoptions parrotRabbitAdoptions) donationRate = 54.27 := by
  sorry

end NUMINAMATH_CALUDE_vet_donation_calculation_l190_19087


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l190_19021

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧
  (1 * a + 3 = 13) ∧
  (3 * b + 1 = 13) ∧
  (∀ (x y : ℕ), x > 3 → y > 3 →
    (1 * x + 3 = 3 * y + 1) →
    (1 * x + 3 ≥ 13)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l190_19021


namespace NUMINAMATH_CALUDE_problem_statement_l190_19020

theorem problem_statement (a b : ℝ) (h : (a + 2)^2 + |b - 1| = 0) :
  (a + b)^2023 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l190_19020


namespace NUMINAMATH_CALUDE_bc_length_l190_19033

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  t.AB = 5 ∧ t.AC = 6 ∧ Real.sin t.A = 3/5

-- Theorem statement
theorem bc_length (t : Triangle) 
  (h_acute : t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2)
  (h_cond : problem_conditions t) : 
  t.BC = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_bc_length_l190_19033


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l190_19002

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a2 : a 2 = 1/2) 
  (h_a5 : a 5 = 4) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l190_19002


namespace NUMINAMATH_CALUDE_cupcake_distribution_l190_19086

/-- Given initial cupcakes, eaten cupcakes, and number of packages, 
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem stating that with 18 initial cupcakes, 8 eaten cupcakes, 
    and 5 packages, there are 2 cupcakes in each package. -/
theorem cupcake_distribution : cupcakes_per_package 18 8 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l190_19086


namespace NUMINAMATH_CALUDE_bell_interval_problem_l190_19017

theorem bell_interval_problem (x : ℕ+) : 
  Nat.lcm x (Nat.lcm 10 (Nat.lcm 14 18)) = 630 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_bell_interval_problem_l190_19017


namespace NUMINAMATH_CALUDE_final_sign_is_minus_l190_19003

/-- Represents the two possible signs on the board -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the board -/
structure Board :=
  (plusCount : Nat)
  (minusCount : Nat)

/-- Applies the transformation rule to two signs -/
def transform (s1 s2 : Sign) : Sign :=
  match s1, s2 with
  | Sign.Plus, Sign.Plus => Sign.Plus
  | Sign.Minus, Sign.Minus => Sign.Plus
  | _, _ => Sign.Minus

/-- Theorem stating that the final sign will be minus -/
theorem final_sign_is_minus 
  (initial : Board)
  (h_initial_plus : initial.plusCount = 2004)
  (h_initial_minus : initial.minusCount = 2005) :
  ∃ (final : Board), final.plusCount + final.minusCount = 1 ∧ final.minusCount = 1 := by
  sorry


end NUMINAMATH_CALUDE_final_sign_is_minus_l190_19003


namespace NUMINAMATH_CALUDE_cake_division_l190_19056

theorem cake_division (pooh_initial piglet_initial : ℚ) : 
  pooh_initial + piglet_initial = 1 →
  piglet_initial + (1/3) * pooh_initial = 3 * piglet_initial →
  pooh_initial = 6/7 ∧ piglet_initial = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_cake_division_l190_19056


namespace NUMINAMATH_CALUDE_square_root_of_nine_l190_19083

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l190_19083


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l190_19091

/-- Represents a five-digit number in the form AB,CBA -/
def abcba (a b c : Nat) : Nat := 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- Check if three digits are distinct -/
def distinct_digits (a b c : Nat) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem greatest_abcba_divisible_by_13 :
  ∀ a b c : Nat,
  a < 10 → b < 10 → c < 10 →
  distinct_digits a b c →
  abcba a b c ≤ 99999 →
  abcba a b c ≡ 0 [MOD 13] →
  abcba a b c ≤ 95159 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l190_19091


namespace NUMINAMATH_CALUDE_square_plus_n_plus_one_is_odd_l190_19053

theorem square_plus_n_plus_one_is_odd (n : ℤ) : Odd (n^2 + n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_plus_n_plus_one_is_odd_l190_19053


namespace NUMINAMATH_CALUDE_width_of_right_triangle_in_square_l190_19034

/-- A right triangle that fits inside a square -/
structure RightTriangleInSquare where
  height : ℝ
  width : ℝ
  square_side : ℝ
  is_right_triangle : True
  fits_in_square : height ≤ square_side ∧ width ≤ square_side

/-- Theorem: The width of a right triangle with height 2 that fits in a 2x2 square is 2 -/
theorem width_of_right_triangle_in_square
  (triangle : RightTriangleInSquare)
  (h_height : triangle.height = 2)
  (h_square : triangle.square_side = 2) :
  triangle.width = 2 :=
sorry

end NUMINAMATH_CALUDE_width_of_right_triangle_in_square_l190_19034


namespace NUMINAMATH_CALUDE_diagonal_angle_tangent_l190_19028

/-- A convex quadrilateral with given properties -/
structure ConvexQuadrilateral where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  convex : Bool

/-- The measure of the acute angle formed by the diagonals -/
def diagonalAngle (q : ConvexQuadrilateral) : ℝ := sorry

/-- Theorem stating the tangent of the diagonal angle -/
theorem diagonal_angle_tangent (q : ConvexQuadrilateral) 
  (h1 : q.area = 30)
  (h2 : q.side1 = 5)
  (h3 : q.side2 = 6)
  (h4 : q.side3 = 9)
  (h5 : q.side4 = 7)
  (h6 : q.convex = true) :
  Real.tan (diagonalAngle q) = 40 / 7 := by sorry

end NUMINAMATH_CALUDE_diagonal_angle_tangent_l190_19028


namespace NUMINAMATH_CALUDE_perp_lines_parallel_perp_planes_parallel_l190_19047

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (linePerpToPlane : Line → Plane → Prop)
variable (planePerpToLine : Plane → Line → Prop)

-- Axioms
axiom distinct_lines (a b : Line) : a ≠ b
axiom distinct_planes (α β : Plane) : α ≠ β

-- Theorem 1
theorem perp_lines_parallel (a b : Line) (α : Plane) :
  linePerpToPlane a α → linePerpToPlane b α → parallelLines a b :=
sorry

-- Theorem 2
theorem perp_planes_parallel (a : Line) (α β : Plane) :
  planePerpToLine α a → planePerpToLine β a → parallelPlanes α β :=
sorry

end NUMINAMATH_CALUDE_perp_lines_parallel_perp_planes_parallel_l190_19047


namespace NUMINAMATH_CALUDE_max_pairs_from_27_l190_19018

theorem max_pairs_from_27 (n : ℕ) (h : n = 27) :
  (n * (n - 1)) / 2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_max_pairs_from_27_l190_19018


namespace NUMINAMATH_CALUDE_airplane_travel_time_l190_19014

/-- Proves that the time taken for an airplane to travel against the wind is 5 hours -/
theorem airplane_travel_time 
  (distance : ℝ) 
  (return_time : ℝ) 
  (still_air_speed : ℝ) 
  (h1 : distance = 3600) 
  (h2 : return_time = 4) 
  (h3 : still_air_speed = 810) : 
  (distance / (still_air_speed - (distance / return_time - still_air_speed))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_airplane_travel_time_l190_19014


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l190_19081

/-- Represents the ages of three brothers -/
structure BrotherAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrotherAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to be proved -/
theorem david_scott_age_difference (ages : BrotherAges) :
  problem_conditions ages → ages.david - ages.scott = 8 := by
  sorry


end NUMINAMATH_CALUDE_david_scott_age_difference_l190_19081


namespace NUMINAMATH_CALUDE_anthonys_pets_l190_19074

theorem anthonys_pets (initial_pets : ℕ) (lost_pets : ℕ) (final_pets : ℕ) :
  initial_pets = 16 →
  lost_pets = 6 →
  final_pets = 8 →
  (initial_pets - lost_pets - final_pets : ℚ) / (initial_pets - lost_pets) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_anthonys_pets_l190_19074


namespace NUMINAMATH_CALUDE_range_of_t_for_right_angle_l190_19082

/-- The theorem stating the range of t for point M(3,t) given the conditions -/
theorem range_of_t_for_right_angle (t : ℝ) : 
  let M : ℝ × ℝ := (3, t)
  let O : ℝ × ℝ := (0, 0)
  let circle_O := {(x, y) : ℝ × ℝ | x^2 + y^2 = 6}
  ∃ (A B : ℝ × ℝ), A ∈ circle_O ∧ B ∈ circle_O ∧ 
    ((M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2) = 0) →
  -Real.sqrt 3 ≤ t ∧ t ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_for_right_angle_l190_19082


namespace NUMINAMATH_CALUDE_systematic_sample_property_l190_19080

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Finset ℕ
  h_class_size : class_size > 0
  h_sample_size : sample_size > 0
  h_sample_size_le : sample_size ≤ class_size
  h_known_seats : known_seats.card < sample_size

/-- The seat number of the missing student in the systematic sample -/
def missing_seat (s : SystematicSample) : ℕ := sorry

/-- Theorem stating the property of the systematic sample -/
theorem systematic_sample_property (s : SystematicSample) 
  (h_seats : s.known_seats = {3, 15, 39, 51}) 
  (h_class_size : s.class_size = 60) 
  (h_sample_size : s.sample_size = 5) : 
  missing_seat s = 27 := by sorry

end NUMINAMATH_CALUDE_systematic_sample_property_l190_19080


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_sum_l190_19088

/-- Given two parallel vectors a and b, prove that tan(α + π/4) = 7 --/
theorem parallel_vectors_tan_sum (α : ℝ) : 
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (Real.sin α, Real.cos α)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →  -- Parallel vectors condition
  Real.tan (α + π/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_sum_l190_19088


namespace NUMINAMATH_CALUDE_complex_expression_equals_100_algebraic_expression_simplification_l190_19099

-- Problem 1
theorem complex_expression_equals_100 :
  (2 * (7 / 9 : ℝ)) ^ (1 / 2 : ℝ) + (1 / 10 : ℝ) ^ (-2 : ℝ) + 
  (2 * (10 / 27 : ℝ)) ^ (-(2 / 3) : ℝ) - 3 * (Real.pi ^ (0 : ℝ)) + 
  (37 / 48 : ℝ) = 100 := by sorry

-- Problem 2
theorem algebraic_expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (a ^ (2 / 3)) * (b ^ (1 / 2))) * (-6 * (a ^ (1 / 2)) * (b ^ (1 / 3))) / 
  (-3 * (a ^ (1 / 6)) * (b ^ (5 / 6))) = 4 * a := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_100_algebraic_expression_simplification_l190_19099


namespace NUMINAMATH_CALUDE_average_weight_increase_l190_19063

theorem average_weight_increase (original_count : ℕ) (original_weight replaced_weight new_weight : ℝ) :
  original_count = 9 →
  replaced_weight = 65 →
  new_weight = 87.5 →
  (new_weight - replaced_weight) / original_count = 2.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l190_19063


namespace NUMINAMATH_CALUDE_olly_shoes_count_l190_19057

/-- The number of shoes needed for Olly's pets -/
def shoes_needed (num_dogs num_cats num_ferrets : ℕ) : ℕ :=
  4 * (num_dogs + num_cats + num_ferrets)

/-- Theorem: Olly needs 24 shoes for his pets -/
theorem olly_shoes_count : shoes_needed 3 2 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_olly_shoes_count_l190_19057


namespace NUMINAMATH_CALUDE_color_film_fraction_l190_19078

/-- Given a committee reviewing films for a festival, this theorem proves
    the fraction of selected films that are in color. -/
theorem color_film_fraction
  (x y : ℕ) -- x and y are natural numbers
  (total_bw : ℕ := 40 * x) -- Total number of black-and-white films
  (total_color : ℕ := 10 * y) -- Total number of color films
  (bw_selected_percent : ℚ := y / x) -- Percentage of black-and-white films selected
  (color_selected_percent : ℚ := 1) -- All color films are selected
  : (total_color : ℚ) / ((bw_selected_percent * total_bw + total_color) : ℚ) = 5 / 26 :=
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l190_19078


namespace NUMINAMATH_CALUDE_poster_area_is_zero_l190_19023

theorem poster_area_is_zero (x y : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (3 * x + 5) * (y + 3) = x * y + 57) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_poster_area_is_zero_l190_19023


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l190_19010

/-- The probability of having a boy or a girl -/
def gender_prob : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having at least one boy and one girl in a family with four children,
    given that the probability of having a boy or a girl is equally likely -/
theorem prob_at_least_one_boy_one_girl (h : gender_prob = 1 / 2) :
  1 - (gender_prob ^ num_children + (1 - gender_prob) ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l190_19010


namespace NUMINAMATH_CALUDE_seniors_physical_books_l190_19046

/-- A survey on book preferences --/
structure BookSurvey where
  total_physical : ℕ
  adults_physical : ℕ
  seniors_ebook : ℕ

/-- The number of seniors preferring physical books --/
def seniors_physical (survey : BookSurvey) : ℕ :=
  survey.total_physical - survey.adults_physical

/-- Theorem: In the given survey, 100 seniors prefer physical books --/
theorem seniors_physical_books (survey : BookSurvey)
  (h1 : survey.total_physical = 180)
  (h2 : survey.adults_physical = 80)
  (h3 : survey.seniors_ebook = 130) :
  seniors_physical survey = 100 := by
  sorry

end NUMINAMATH_CALUDE_seniors_physical_books_l190_19046


namespace NUMINAMATH_CALUDE_hendecagon_diagonals_l190_19066

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hendecagon is an 11-sided polygon -/
def hendecagon_sides : ℕ := 11

/-- The number of diagonals in a hendecagon is 44 -/
theorem hendecagon_diagonals : num_diagonals hendecagon_sides = 44 := by
  sorry

end NUMINAMATH_CALUDE_hendecagon_diagonals_l190_19066


namespace NUMINAMATH_CALUDE_cups_per_girl_l190_19075

theorem cups_per_girl (total_students : Nat) (boys : Nat) (cups_per_boy : Nat) (total_cups : Nat)
  (h1 : total_students = 30)
  (h2 : boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_cups = 90)
  (h5 : boys * 2 = total_students - boys) :
  (total_cups - boys * cups_per_boy) / (total_students - boys) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cups_per_girl_l190_19075


namespace NUMINAMATH_CALUDE_min_cost_2009_l190_19050

/-- Represents the denominations of coins available --/
inductive Coin
  | One
  | Two
  | Five
  | Ten

/-- Represents an arithmetic expression --/
inductive Expr
  | Const (n : ℕ)
  | Add (e1 e2 : Expr)
  | Sub (e1 e2 : Expr)
  | Mul (e1 e2 : Expr)
  | Div (e1 e2 : Expr)

/-- Evaluates an expression to a natural number --/
def eval : Expr → ℕ
  | Expr.Const n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Sub e1 e2 => eval e1 - eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2
  | Expr.Div e1 e2 => eval e1 / eval e2

/-- Calculates the cost of an expression in rubles --/
def cost : Expr → ℕ
  | Expr.Const n => n
  | Expr.Add e1 e2 => cost e1 + cost e2
  | Expr.Sub e1 e2 => cost e1 + cost e2
  | Expr.Mul e1 e2 => cost e1 + cost e2
  | Expr.Div e1 e2 => cost e1 + cost e2

/-- Theorem: The minimum cost to create an expression equal to 2009 is 23 rubles --/
theorem min_cost_2009 :
  ∃ (e : Expr), eval e = 2009 ∧ cost e = 23 ∧
  (∀ (e' : Expr), eval e' = 2009 → cost e' ≥ 23) :=
sorry


end NUMINAMATH_CALUDE_min_cost_2009_l190_19050


namespace NUMINAMATH_CALUDE_range_of_a_l190_19039

def A := {x : ℝ | -1 < x ∧ x < 6}
def B (a : ℝ) := {x : ℝ | x^2 - 2*x + 1 - a^2 ≥ 0}

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∉ A → x ∈ B a) ∧ (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → 
  0 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l190_19039


namespace NUMINAMATH_CALUDE_rectangle_max_area_rectangle_max_area_value_l190_19044

/-- Represents a rectangle with length, width, and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  perimeterConstraint : perimeter = 2 * (length + width)

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: The area of a rectangle with fixed perimeter is maximized when it's a square -/
theorem rectangle_max_area (p : ℝ) (hp : p > 0) :
  ∃ (r : Rectangle), r.perimeter = p ∧
    ∀ (s : Rectangle), s.perimeter = p → r.area ≥ s.area ∧
    r.length = p / 4 ∧ r.width = p / 4 :=
  sorry

/-- Corollary: The maximum area of a rectangle with perimeter p is p^2 / 16 -/
theorem rectangle_max_area_value (p : ℝ) (hp : p > 0) :
  ∃ (r : Rectangle), r.perimeter = p ∧
    ∀ (s : Rectangle), s.perimeter = p → r.area ≥ s.area ∧
    r.area = p^2 / 16 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_rectangle_max_area_value_l190_19044


namespace NUMINAMATH_CALUDE_same_color_probability_is_seven_ninths_l190_19011

/-- Represents a die with a specific number of sides and color distribution -/
structure Die where
  sides : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  valid : red + blue + green = sides

/-- Calculate the probability of two dice showing the same color -/
def same_color_probability (d1 d2 : Die) : ℚ :=
  let p_red := (d1.red : ℚ) / d1.sides * (d2.red : ℚ) / d2.sides
  let p_blue := (d1.blue : ℚ) / d1.sides * (d2.blue : ℚ) / d2.sides
  let p_green := (d1.green : ℚ) / d1.sides * (d2.green : ℚ) / d2.sides
  p_red + p_blue + p_green

/-- The first die with 12 sides: 3 red, 4 blue, 5 green -/
def die1 : Die := {
  sides := 12,
  red := 3,
  blue := 4,
  green := 5,
  valid := by simp
}

/-- The second die with 15 sides: 5 red, 3 blue, 7 green -/
def die2 : Die := {
  sides := 15,
  red := 5,
  blue := 3,
  green := 7,
  valid := by simp
}

/-- Theorem stating that the probability of both dice showing the same color is 7/9 -/
theorem same_color_probability_is_seven_ninths :
  same_color_probability die1 die2 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_seven_ninths_l190_19011


namespace NUMINAMATH_CALUDE_square_root_of_sqrt_16_l190_19030

theorem square_root_of_sqrt_16 : 
  {x : ℝ | x^2 = Real.sqrt 16} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sqrt_16_l190_19030


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l190_19095

theorem quadratic_function_k_value (a b c : ℤ) (k : ℤ) : 
  let f : ℝ → ℝ := λ x => (a * x^2 + b * x + c : ℝ)
  (f 1 = 0) →
  (60 < f 9 ∧ f 9 < 70) →
  (90 < f 10 ∧ f 10 < 100) →
  (10000 * k < f 100 ∧ f 100 < 10000 * (k + 1)) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l190_19095


namespace NUMINAMATH_CALUDE_currency_notes_count_l190_19019

theorem currency_notes_count (total_amount : ℕ) (amount_in_50 : ℕ) (denomination_50 : ℕ) (denomination_100 : ℕ) :
  total_amount = 5000 →
  amount_in_50 = 3500 →
  denomination_50 = 50 →
  denomination_100 = 100 →
  (amount_in_50 / denomination_50 + (total_amount - amount_in_50) / denomination_100 : ℕ) = 85 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_count_l190_19019


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l190_19012

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, 2 * S n = a n * (a n + 1)) :
  (∀ n, a n = n) ∧ (∀ n, a (n + 1) - a n = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l190_19012


namespace NUMINAMATH_CALUDE_output_value_S_l190_19032

theorem output_value_S : ∃ S : ℕ, S = 1 * 3^1 + 2 * 3^2 + 3 * 3^3 ∧ S = 102 := by
  sorry

end NUMINAMATH_CALUDE_output_value_S_l190_19032


namespace NUMINAMATH_CALUDE_mixed_groups_count_l190_19038

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 + boy_boy_photos + girl_girl_photos = total_groups * group_size :=
by sorry

end NUMINAMATH_CALUDE_mixed_groups_count_l190_19038


namespace NUMINAMATH_CALUDE_digit_150_of_17_150_l190_19068

/-- The decimal representation of 17/150 -/
def decimal_rep : ℚ := 17 / 150

/-- The nth digit after the decimal point in a rational number -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_150_of_17_150 :
  nth_digit decimal_rep 150 = 3 :=
sorry

end NUMINAMATH_CALUDE_digit_150_of_17_150_l190_19068


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l190_19070

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l190_19070


namespace NUMINAMATH_CALUDE_greaterThanOne_is_random_event_l190_19055

-- Define the type for outcomes of rolling a die
def DieOutcome := Fin 6

-- Define the event "greater than 1"
def greaterThanOne (outcome : DieOutcome) : Prop := outcome.val > 1

-- Define what it means for an event to be random
def isRandomEvent (event : DieOutcome → Prop) : Prop :=
  ∃ (o1 o2 : DieOutcome), event o1 ∧ ¬event o2

-- Theorem stating that "greater than 1" is a random event
theorem greaterThanOne_is_random_event : isRandomEvent greaterThanOne := by
  sorry


end NUMINAMATH_CALUDE_greaterThanOne_is_random_event_l190_19055
