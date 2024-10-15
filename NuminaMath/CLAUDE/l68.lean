import Mathlib

namespace NUMINAMATH_CALUDE_inequality_condition_l68_6838

theorem inequality_condition (A B C : ℝ) : 
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔ 
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l68_6838


namespace NUMINAMATH_CALUDE_toothpaste_duration_l68_6880

/-- Represents the amount of toothpaste in grams --/
def toothpasteAmount : ℝ := 105

/-- Represents the amount of toothpaste used by Anne's dad per brushing --/
def dadUsage : ℝ := 3

/-- Represents the amount of toothpaste used by Anne's mom per brushing --/
def momUsage : ℝ := 2

/-- Represents the amount of toothpaste used by Anne per brushing --/
def anneUsage : ℝ := 1

/-- Represents the amount of toothpaste used by Anne's brother per brushing --/
def brotherUsage : ℝ := 1

/-- Represents the number of times each family member brushes their teeth per day --/
def brushingsPerDay : ℕ := 3

/-- Theorem stating that the toothpaste will last for 5 days --/
theorem toothpaste_duration : 
  ∃ (days : ℝ), days = 5 ∧ 
  days * (dadUsage + momUsage + anneUsage + brotherUsage) * brushingsPerDay = toothpasteAmount :=
by sorry

end NUMINAMATH_CALUDE_toothpaste_duration_l68_6880


namespace NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l68_6881

theorem arithmetic_sequence_logarithm (a b : ℝ) (m : ℝ) :
  a > 0 ∧ b > 0 ∧
  (2 : ℝ) ^ a = m ∧
  (3 : ℝ) ^ b = m ∧
  2 * a * b = a + b →
  m = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_logarithm_l68_6881


namespace NUMINAMATH_CALUDE_clothing_popularity_l68_6858

/-- Represents the sales of clothing on a given day in July -/
def sales (n : ℕ) : ℕ :=
  if n ≤ 13 then 3 * n else 65 - 2 * n

/-- Represents the cumulative sales up to a given day in July -/
def cumulative_sales (n : ℕ) : ℕ :=
  if n ≤ 13 then (3 + 3 * n) * n / 2 else 273 + (51 - n) * (n - 13)

/-- The day when the clothing becomes popular -/
def popular_start : ℕ := 12

/-- The day when the clothing is no longer popular -/
def popular_end : ℕ := 22

theorem clothing_popularity :
  (∀ n : ℕ, n ≥ popular_start → n ≤ popular_end → cumulative_sales n ≥ 200) ∧
  (∀ n : ℕ, n > popular_end → sales n < 20) ∧
  popular_end - popular_start + 1 = 11 := by sorry

end NUMINAMATH_CALUDE_clothing_popularity_l68_6858


namespace NUMINAMATH_CALUDE_minimum_time_two_people_one_bicycle_l68_6895

/-- The minimum time problem for two people traveling with one bicycle -/
theorem minimum_time_two_people_one_bicycle
  (distance : ℝ)
  (walk_speed1 walk_speed2 bike_speed1 bike_speed2 : ℝ)
  (h_distance : distance = 40)
  (h_walk_speed1 : walk_speed1 = 4)
  (h_walk_speed2 : walk_speed2 = 6)
  (h_bike_speed1 : bike_speed1 = 30)
  (h_bike_speed2 : bike_speed2 = 20)
  (h_positive : walk_speed1 > 0 ∧ walk_speed2 > 0 ∧ bike_speed1 > 0 ∧ bike_speed2 > 0) :
  ∃ (t : ℝ), t = 25/9 ∧ 
  ∀ (t' : ℝ), (∃ (x y : ℝ), 
    x ≥ 0 ∧ y ≥ 0 ∧
    bike_speed1 * x + walk_speed1 * y = distance ∧
    walk_speed2 * x + bike_speed2 * y = distance ∧
    t' = x + y) → t ≤ t' :=
by sorry

end NUMINAMATH_CALUDE_minimum_time_two_people_one_bicycle_l68_6895


namespace NUMINAMATH_CALUDE_odd_functions_properties_l68_6837

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 + x - k
def g (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem odd_functions_properties :
  (∀ x, g (-x) = -g x) ∧  -- g is an odd function
  (g 1 = -2) ∧  -- g achieves minimum -2 at x = 1
  (∀ x, g x ≤ 2) ∧  -- maximum value of g is 2
  (∀ k, (∀ x ∈ Set.Icc (-1) 3, f k x ≤ g x) → k ≥ 8) ∧  -- range of k for f ≤ g on [-1,3]
  (∀ k, (∀ x₁ ∈ Set.Icc (-1) 3, ∀ x₂ ∈ Set.Icc (-1) 3, f k x₁ ≤ g x₂) → k ≥ 23)  -- range of k for f(x₁) ≤ g(x₂)
  := by sorry

end NUMINAMATH_CALUDE_odd_functions_properties_l68_6837


namespace NUMINAMATH_CALUDE_missing_number_is_four_l68_6892

/-- The structure of the problem -/
structure BoxStructure where
  top_left : ℕ
  top_right : ℕ
  middle_left : ℕ
  middle_right : ℕ
  bottom : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (b : BoxStructure) : Prop :=
  b.middle_left = b.top_left * b.top_right ∧
  b.bottom = b.middle_left * b.middle_right ∧
  b.middle_left = 30 ∧
  b.top_left = 6 ∧
  b.top_right = 5 ∧
  b.bottom = 600

/-- The theorem to prove -/
theorem missing_number_is_four :
  ∀ b : BoxStructure, satisfies_conditions b → b.middle_right = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_four_l68_6892


namespace NUMINAMATH_CALUDE_mn_value_l68_6809

theorem mn_value (m n : ℤ) : 
  (∀ x : ℤ, x^2 + m*x - 15 = (x + 3)*(x + n)) → m*n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l68_6809


namespace NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l68_6845

theorem cos_pi_third_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.cos (π / 3 + 2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l68_6845


namespace NUMINAMATH_CALUDE_correct_marble_distribution_l68_6891

/-- Represents the number of marbles each boy has -/
structure MarbleDistribution where
  middle : ℕ
  least : ℕ
  most : ℕ

/-- Checks if the given marble distribution satisfies the problem conditions -/
def is_valid_distribution (d : MarbleDistribution) : Prop :=
  -- The ratio of marbles is 4:2:3
  4 * d.middle = 2 * d.most ∧
  2 * d.least = 3 * d.middle ∧
  -- The boy with the least marbles has 10 more than twice the middle boy's marbles
  d.least = 2 * d.middle + 10 ∧
  -- The total number of marbles is 156
  d.middle + d.least + d.most = 156

/-- The theorem stating the correct distribution of marbles -/
theorem correct_marble_distribution :
  is_valid_distribution ⟨23, 57, 76⟩ := by sorry

end NUMINAMATH_CALUDE_correct_marble_distribution_l68_6891


namespace NUMINAMATH_CALUDE_cylinder_volume_unchanged_l68_6808

/-- Theorem: For a cylinder with radius 5 inches and height 4 inches, 
    the value of x that keeps the volume unchanged when the radius 
    is increased by x and the height is decreased by x is 5 - 2√10. -/
theorem cylinder_volume_unchanged (R H : ℝ) (x : ℝ) : 
  R = 5 → H = 4 → 
  π * R^2 * H = π * (R + x)^2 * (H - x) → 
  x = 5 - 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_unchanged_l68_6808


namespace NUMINAMATH_CALUDE_frog_to_hamster_ratio_l68_6888

-- Define the lifespans of the animals
def bat_lifespan : ℕ := 10
def hamster_lifespan : ℕ := bat_lifespan - 6

-- Define the total lifespan
def total_lifespan : ℕ := 30

-- Define the frog's lifespan as a function of the hamster's
def frog_lifespan : ℕ := total_lifespan - (bat_lifespan + hamster_lifespan)

-- Theorem to prove
theorem frog_to_hamster_ratio :
  frog_lifespan / hamster_lifespan = 4 :=
by sorry

end NUMINAMATH_CALUDE_frog_to_hamster_ratio_l68_6888


namespace NUMINAMATH_CALUDE_school_commute_time_l68_6870

theorem school_commute_time (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_time > 0 →
  usual_rate > 0 →
  (6 / 7 * usual_rate) * (usual_time - 4) = usual_rate * usual_time →
  usual_time = 28 := by
sorry

end NUMINAMATH_CALUDE_school_commute_time_l68_6870


namespace NUMINAMATH_CALUDE_ny_mets_fans_count_l68_6863

theorem ny_mets_fans_count (total_fans : ℕ) (yankees_mets_ratio : ℚ) (mets_redsox_ratio : ℚ) :
  total_fans = 390 →
  yankees_mets_ratio = 3 / 2 →
  mets_redsox_ratio = 4 / 5 →
  ∃ (yankees mets redsox : ℕ),
    yankees + mets + redsox = total_fans ∧
    (yankees : ℚ) / mets = yankees_mets_ratio ∧
    (mets : ℚ) / redsox = mets_redsox_ratio ∧
    mets = 104 :=
by sorry

end NUMINAMATH_CALUDE_ny_mets_fans_count_l68_6863


namespace NUMINAMATH_CALUDE_subsequence_appears_l68_6825

/-- Defines the sequence where each digit after the first four is the last digit of the sum of the previous four digits -/
def digit_sequence : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| n + 4 => (digit_sequence n + digit_sequence (n + 1) + digit_sequence (n + 2) + digit_sequence (n + 3)) % 10

/-- Checks if the subsequence 8123 appears starting at position n in the sequence -/
def appears_at (n : ℕ) : Prop :=
  digit_sequence n = 8 ∧
  digit_sequence (n + 1) = 1 ∧
  digit_sequence (n + 2) = 2 ∧
  digit_sequence (n + 3) = 3

/-- Theorem stating that the subsequence 8123 appears in the sequence -/
theorem subsequence_appears : ∃ n : ℕ, appears_at n := by
  sorry

end NUMINAMATH_CALUDE_subsequence_appears_l68_6825


namespace NUMINAMATH_CALUDE_equation_has_solution_l68_6846

theorem equation_has_solution (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_solution_l68_6846


namespace NUMINAMATH_CALUDE_negative_five_meters_decrease_l68_6815

-- Define a type for distance changes
inductive DistanceChange
| Increase (amount : ℤ)
| Decrease (amount : ℤ)

-- Define a function to interpret integers as distance changes
def interpretDistance (d : ℤ) : DistanceChange :=
  if d > 0 then DistanceChange.Increase d
  else DistanceChange.Decrease (-d)

-- Theorem statement
theorem negative_five_meters_decrease :
  interpretDistance (-5) = DistanceChange.Decrease 5 :=
by sorry

end NUMINAMATH_CALUDE_negative_five_meters_decrease_l68_6815


namespace NUMINAMATH_CALUDE_nonzero_real_equation_solution_l68_6804

theorem nonzero_real_equation_solution (x : ℝ) (h : x ≠ 0) :
  (5 * x)^10 = (10 * x)^5 ↔ x = 2/5 := by sorry

end NUMINAMATH_CALUDE_nonzero_real_equation_solution_l68_6804


namespace NUMINAMATH_CALUDE_european_stamps_cost_l68_6856

/-- Represents a country with its stamp counts and price --/
structure Country where
  name : String
  price : ℚ
  count_80s : ℕ
  count_90s : ℕ

/-- Calculates the total cost of stamps for a country in both decades --/
def totalCost (c : Country) : ℚ :=
  c.price * (c.count_80s + c.count_90s)

/-- The set of European countries in Laura's collection --/
def europeanCountries : List Country :=
  [{ name := "France", price := 9/100, count_80s := 10, count_90s := 12 },
   { name := "Spain", price := 7/100, count_80s := 18, count_90s := 16 }]

theorem european_stamps_cost :
  List.sum (europeanCountries.map totalCost) = 436/100 := by
  sorry

end NUMINAMATH_CALUDE_european_stamps_cost_l68_6856


namespace NUMINAMATH_CALUDE_tan_half_period_l68_6860

/-- The period of tan(x/2) is 2π -/
theorem tan_half_period : 
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.tan (x / 2)) → 
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = 2 * Real.pi := by
  sorry

/-- The period of tan(x) is π -/
axiom tan_period : 
  ∀ x : ℝ, Real.tan (x + Real.pi) = Real.tan x

end NUMINAMATH_CALUDE_tan_half_period_l68_6860


namespace NUMINAMATH_CALUDE_average_speed_swim_run_l68_6865

/-- 
Given a swimmer who swims at 1 mile per hour and runs at 11 miles per hour,
their average speed for these two events (assuming equal distances for both)
is 11/6 miles per hour.
-/
theorem average_speed_swim_run :
  let swim_speed : ℝ := 1
  let run_speed : ℝ := 11
  let total_distance : ℝ := 2 -- Assuming 1 mile each for swimming and running
  let swim_time : ℝ := 1 -- Time to swim 1 mile at 1 mph
  let run_time : ℝ := 1 / 11 -- Time to run 1 mile at 11 mph
  let total_time : ℝ := swim_time + run_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 11 / 6 := by sorry

end NUMINAMATH_CALUDE_average_speed_swim_run_l68_6865


namespace NUMINAMATH_CALUDE_remaining_area_after_triangles_cut_l68_6849

theorem remaining_area_after_triangles_cut (grid_side : ℕ) (dark_rect_dim : ℕ × ℕ) (light_rect_dim : ℕ × ℕ) : 
  grid_side = 6 →
  dark_rect_dim = (1, 3) →
  light_rect_dim = (2, 3) →
  (grid_side^2 : ℝ) - (dark_rect_dim.1 * dark_rect_dim.2 + light_rect_dim.1 * light_rect_dim.2 : ℝ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_triangles_cut_l68_6849


namespace NUMINAMATH_CALUDE_sundress_price_problem_l68_6802

theorem sundress_price_problem (P : ℝ) : 
  P - (P * 0.85 * 1.25) = 4.5 → P * 0.85 = 61.2 := by
  sorry

end NUMINAMATH_CALUDE_sundress_price_problem_l68_6802


namespace NUMINAMATH_CALUDE_max_coach_handshakes_zero_l68_6884

/-- The total number of handshakes in the tournament -/
def total_handshakes : ℕ := 465

/-- The number of players in the tournament -/
def num_players : ℕ := 31

/-- The number of handshakes between players -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of handshakes the coach participated in -/
def coach_handshakes : ℕ := total_handshakes - player_handshakes num_players

theorem max_coach_handshakes_zero :
  coach_handshakes = 0 ∧ 
  ∀ n : ℕ, n > num_players → player_handshakes n > total_handshakes := by
  sorry


end NUMINAMATH_CALUDE_max_coach_handshakes_zero_l68_6884


namespace NUMINAMATH_CALUDE_weekly_reading_time_l68_6801

def daily_meditation_time : ℝ := 1
def daily_reading_time : ℝ := 2 * daily_meditation_time
def days_in_week : ℕ := 7

theorem weekly_reading_time :
  daily_reading_time * (days_in_week : ℝ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_weekly_reading_time_l68_6801


namespace NUMINAMATH_CALUDE_cubic_roots_fourth_power_sum_l68_6899

theorem cubic_roots_fourth_power_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 2*x^2 + 3*x - 4 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  a^4 + b^4 + c^4 = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_fourth_power_sum_l68_6899


namespace NUMINAMATH_CALUDE_probability_even_sum_l68_6862

theorem probability_even_sum (wheel1_even : ℚ) (wheel1_odd : ℚ) 
  (wheel2_even : ℚ) (wheel2_odd : ℚ) : 
  wheel1_even = 1/4 →
  wheel1_odd = 3/4 →
  wheel2_even = 2/3 →
  wheel2_odd = 1/3 →
  wheel1_even + wheel1_odd = 1 →
  wheel2_even + wheel2_odd = 1 →
  wheel1_even * wheel2_even + wheel1_odd * wheel2_odd = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l68_6862


namespace NUMINAMATH_CALUDE_square_area_in_triangle_l68_6833

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A square in a 2D plane -/
structure Square where
  corners : Fin 4 → ℝ × ℝ

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The area of a square -/
def Square.area (s : Square) : ℝ := sorry

/-- Predicate to check if a square lies within a triangle -/
def Square.liesWithin (s : Square) (t : Triangle) : Prop := sorry

/-- Theorem: The area of any square lying within a triangle does not exceed half of the area of that triangle -/
theorem square_area_in_triangle (t : Triangle) (s : Square) :
  s.liesWithin t → s.area ≤ (1/2) * t.area := by sorry

end NUMINAMATH_CALUDE_square_area_in_triangle_l68_6833


namespace NUMINAMATH_CALUDE_vacant_seats_l68_6890

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 62/100) : 
  (total_seats : ℚ) * (1 - filled_percentage) = 228 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l68_6890


namespace NUMINAMATH_CALUDE_like_terms_exponent_equality_l68_6810

theorem like_terms_exponent_equality (a b : ℝ) (m : ℝ) : 
  (∃ k : ℝ, -2 * a^(2-m) * b^3 = k * (-2 * a^(4-3*m) * b^3)) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_equality_l68_6810


namespace NUMINAMATH_CALUDE_circle_area_l68_6800

theorem circle_area (r : ℝ) (h : 6 * (1 / (2 * Real.pi * r)) = r) : π * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l68_6800


namespace NUMINAMATH_CALUDE_gadget_sales_sum_l68_6861

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Gadget sales problem -/
theorem gadget_sales_sum :
  arithmetic_sum 2 3 25 = 950 := by
  sorry

end NUMINAMATH_CALUDE_gadget_sales_sum_l68_6861


namespace NUMINAMATH_CALUDE_sprite_volume_l68_6851

def maazaVolume : ℕ := 80
def pepsiVolume : ℕ := 144
def totalCans : ℕ := 37

def canVolume : ℕ := Nat.gcd maazaVolume pepsiVolume

theorem sprite_volume :
  ∃ (spriteVolume : ℕ),
    spriteVolume = canVolume * (totalCans - (maazaVolume / canVolume + pepsiVolume / canVolume)) ∧
    spriteVolume = 368 := by
  sorry

end NUMINAMATH_CALUDE_sprite_volume_l68_6851


namespace NUMINAMATH_CALUDE_integer_solution_for_inequalities_l68_6839

theorem integer_solution_for_inequalities : 
  ∃! (n : ℤ), n + 15 > 16 ∧ -3 * n^2 > -27 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_solution_for_inequalities_l68_6839


namespace NUMINAMATH_CALUDE_parabola_above_line_l68_6883

theorem parabola_above_line (a : ℝ) : 
  (∀ x ∈ Set.Icc a (a + 1), x^2 - a*x + 3 > 9/4) ↔ a > -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_above_line_l68_6883


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l68_6816

/-- The equations of the asymptotes of the hyperbola y²/9 - x²/4 = 1 are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := λ x y => y^2/9 - x^2/4 - 1
  ∀ x y : ℝ, h x y = 0 →
  ∃ k : ℝ, k = 3/2 ∧
  (∀ ε > 0, ∃ M : ℝ, ∀ x y : ℝ, h x y = 0 ∧ |x| > M →
    (|y - k*x| < ε*|x| ∨ |y + k*x| < ε*|x|)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l68_6816


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l68_6822

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def OnLine (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

def AngleBisector (A B C : Point) (L : Line) : Prop :=
  -- This is a simplified definition and may need to be expanded
  OnLine A L ∧ OnLine B L

-- The main theorem
theorem triangle_construction_theorem :
  ∀ (N M : Point) (l : Line),
  ∃ (A B C : Point),
    Midpoint A C N ∧
    Midpoint B C M ∧
    AngleBisector A B C l :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l68_6822


namespace NUMINAMATH_CALUDE_calvin_winning_condition_l68_6811

/-- The game state represents the current configuration of coins on the circle. -/
structure GameState where
  n : ℕ
  coins : Fin (2 * n + 1) → Bool

/-- A player's move in the game. -/
inductive Move
  | calvin : Fin (2 * n + 1) → Move
  | hobbes : Option (Fin (2 * n + 1)) → Move

/-- Applies a move to the current game state. -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Counts the number of tails in the current game state. -/
def countTails (state : GameState) : ℕ :=
  sorry

/-- Determines if a player has a winning strategy for the game. -/
def hasWinningStrategy (n k : ℕ) (player : Bool) : Prop :=
  sorry

/-- The main theorem stating the conditions for Calvin's victory. -/
theorem calvin_winning_condition (n k : ℕ) (h1 : n > 1) (h2 : k ≥ 1) :
  hasWinningStrategy n k true ↔ k ≤ n + 1 :=
  sorry

end NUMINAMATH_CALUDE_calvin_winning_condition_l68_6811


namespace NUMINAMATH_CALUDE_range_of_f_inequality_l68_6872

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem range_of_f_inequality 
  (hdom : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → f x ≠ 0 → True)
  (hderiv : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → HasDerivAt f (x^2 + 2 * Real.cos x) x)
  (hf0 : f 0 = 0) :
  {x : ℝ | f (1 + x) + f (x - x^2) > 0} = Set.Ioo (1 - Real.sqrt 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_inequality_l68_6872


namespace NUMINAMATH_CALUDE_not_always_divisible_by_19_l68_6854

theorem not_always_divisible_by_19 : ∃ (a b : ℤ), ¬(19 ∣ ((3*a + 2)^3 - (3*b + 2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_not_always_divisible_by_19_l68_6854


namespace NUMINAMATH_CALUDE_nickel_probability_l68_6875

/-- Represents the types of coins in the box -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the box in cents -/
def total_value : Coin → ℕ
  | Coin.Dime => 500
  | Coin.Nickel => 250
  | Coin.Penny => 100

/-- The number of coins of each type in the box -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the box -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly selecting a nickel from the box -/
theorem nickel_probability : 
  (coin_count Coin.Nickel : ℚ) / total_coins = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_nickel_probability_l68_6875


namespace NUMINAMATH_CALUDE_dot_product_range_l68_6866

/-- The locus M is defined as the set of points (x, y) satisfying x²/3 + y² = 1 -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

/-- F is the point (2, 0) -/
def F : ℝ × ℝ := (2, 0)

/-- Given two points on M, compute their dot product with respect to F -/
def dot_product_with_F (C D : ℝ × ℝ) : ℝ :=
  let FC := (C.1 - F.1, C.2 - F.2)
  let FD := (D.1 - F.1, D.2 - F.2)
  FC.1 * FD.1 + FC.2 * FD.2

/-- The main theorem stating the range of the dot product -/
theorem dot_product_range (C D : ℝ × ℝ) (hC : C ∈ M) (hD : D ∈ M) 
  (h_line : ∃ (k : ℝ), C.2 = k * (C.1 - 2) ∧ D.2 = k * (D.1 - 2)) :
  1/3 < dot_product_with_F C D ∧ dot_product_with_F C D ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_dot_product_range_l68_6866


namespace NUMINAMATH_CALUDE_matrix_max_min_element_l68_6889

theorem matrix_max_min_element
  (m n : ℕ)
  (a : Fin m → ℝ)
  (b : Fin n → ℝ)
  (p : Fin m → ℝ)
  (q : Fin n → ℝ)
  (hp : ∀ i, p i > 0)
  (hq : ∀ j, q j > 0) :
  ∃ (k : Fin m) (l : Fin n),
    (∀ j, (a k + b l) / (p k + q l) ≥ (a k + b j) / (p k + q j)) ∧
    (∀ i, (a k + b l) / (p k + q l) ≤ (a i + b l) / (p i + q l)) :=
by sorry

end NUMINAMATH_CALUDE_matrix_max_min_element_l68_6889


namespace NUMINAMATH_CALUDE_decreasing_order_l68_6813

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the even property of f(x-1)
axiom f_even : ∀ x : ℝ, f (-x - 1) = f (x - 1)

-- Define the decreasing property of f on [-1,+∞)
axiom f_decreasing : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ ≥ -1 → x₂ ≥ -1 → 
  (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log (7/2) / Real.log (1/2))
noncomputable def b : ℝ := f (Real.log (7/2) / Real.log (1/3))
noncomputable def c : ℝ := f (Real.log (3/2) / Real.log 2)

-- The theorem to prove
theorem decreasing_order : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_decreasing_order_l68_6813


namespace NUMINAMATH_CALUDE_xiaoming_age_proof_l68_6871

/-- Xiaoming's current age -/
def xiaoming_age : ℕ := 6

/-- The current age of each of Xiaoming's younger brothers -/
def brother_age : ℕ := 2

/-- The number of Xiaoming's younger brothers -/
def num_brothers : ℕ := 3

/-- Years into the future for the second condition -/
def future_years : ℕ := 6

theorem xiaoming_age_proof :
  (xiaoming_age = num_brothers * brother_age) ∧
  (num_brothers * (brother_age + future_years) = 2 * (xiaoming_age + future_years)) →
  xiaoming_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_age_proof_l68_6871


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l68_6842

/-- Represents the price reduction scenario for a mobile phone -/
def price_reduction (original_price final_price x : ℝ) : Prop :=
  original_price * (1 - x)^2 = final_price

/-- Theorem stating the correct equation for the given price reduction scenario -/
theorem correct_price_reduction_equation :
  ∃ x : ℝ, price_reduction 1185 580 x :=
sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l68_6842


namespace NUMINAMATH_CALUDE_x_is_even_l68_6897

theorem x_is_even (x : ℤ) (h : ∃ (k : ℤ), (2 * x) / 3 - x / 6 = k) : ∃ (m : ℤ), x = 2 * m := by
  sorry

end NUMINAMATH_CALUDE_x_is_even_l68_6897


namespace NUMINAMATH_CALUDE_cos_270_degrees_l68_6841

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l68_6841


namespace NUMINAMATH_CALUDE_trig_problem_l68_6868

theorem trig_problem (α β : Real) 
  (h1 : Real.cos (α - β/2) = -2 * Real.sqrt 7 / 7)
  (h2 : Real.sin (α/2 - β) = 1/2)
  (h3 : π/2 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π/2) :
  Real.cos ((α + β)/2) = -Real.sqrt 21 / 14 ∧ 
  Real.tan (α + β) = 5 * Real.sqrt 3 / 11 := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l68_6868


namespace NUMINAMATH_CALUDE_donation_problem_l68_6840

theorem donation_problem (first_total second_total : ℕ) 
  (donor_ratio : ℚ) (avg_diff : ℕ) :
  first_total = 60000 →
  second_total = 150000 →
  donor_ratio = 3/2 →
  avg_diff = 20 →
  ∃ (first_donors : ℕ),
    first_donors = 2000 ∧
    (donor_ratio * first_donors : ℚ) = 3000 ∧
    (second_total : ℚ) / (donor_ratio * first_donors) - 
    (first_total : ℚ) / first_donors = avg_diff :=
by sorry

end NUMINAMATH_CALUDE_donation_problem_l68_6840


namespace NUMINAMATH_CALUDE_paper_remaining_l68_6836

theorem paper_remaining (total : ℕ) (used : ℕ) (h1 : total = 900) (h2 : used = 156) :
  total - used = 744 := by
  sorry

end NUMINAMATH_CALUDE_paper_remaining_l68_6836


namespace NUMINAMATH_CALUDE_trigonometric_identities_l68_6852

theorem trigonometric_identities :
  (2 * Real.sin (30 * π / 180) - Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1 / 2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + 
   Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l68_6852


namespace NUMINAMATH_CALUDE_min_value_bisecting_line_l68_6844

/-- The minimum value of 1/a + 1/b for a line ax + by - 1 = 0 bisecting a specific circle -/
theorem min_value_bisecting_line (a b : ℝ) : 
  a * b > 0 → 
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 - 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + 1 = 0 → (a * x + b * y - 1) * (a * x + b * y - 1) ≤ (a^2 + b^2) * ((x-1)^2 + (y-2)^2)) →
  (1/a + 1/b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_bisecting_line_l68_6844


namespace NUMINAMATH_CALUDE_unique_solution_count_l68_6823

/-- A system of equations has exactly one solution -/
def has_unique_solution (k : ℝ) : Prop :=
  ∃! x y : ℝ, x^2 + y^2 = 2*k^2 ∧ k*x - y = 2*k

/-- The number of real values of k for which the system has a unique solution -/
theorem unique_solution_count :
  ∃ S : Finset ℝ, (∀ k : ℝ, k ∈ S ↔ has_unique_solution k) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_count_l68_6823


namespace NUMINAMATH_CALUDE_element_in_set_l68_6843

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l68_6843


namespace NUMINAMATH_CALUDE_other_coin_denomination_l68_6834

/-- Given a total of 336 coins with a total value of 7100 paise,
    where 260 of the coins are 20 paise coins,
    prove that the denomination of the other type of coin is 25 paise. -/
theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h_total_coins : total_coins = 336)
  (h_total_value : total_value = 7100)
  (h_twenty_paise_coins : twenty_paise_coins = 260) :
  let other_coins := total_coins - twenty_paise_coins
  let other_denomination := (total_value - 20 * twenty_paise_coins) / other_coins
  other_denomination = 25 :=
by sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l68_6834


namespace NUMINAMATH_CALUDE_cylindrical_tank_volume_increase_l68_6857

theorem cylindrical_tank_volume_increase (R H : ℝ) (hR : R = 10) (hH : H = 5) :
  ∃ k : ℝ, k > 0 ∧
  (π * (k * R)^2 * H - π * R^2 * H = π * R^2 * (H + k) - π * R^2 * H) ∧
  k = (1 + Real.sqrt 101) / 10 := by
sorry

end NUMINAMATH_CALUDE_cylindrical_tank_volume_increase_l68_6857


namespace NUMINAMATH_CALUDE_plane_sphere_ratio_sum_l68_6882

/-- Given a plane passing through (a,b,c) and intersecting the coordinate axes, 
    prove that the sum of ratios of the fixed point coordinates to the sphere center coordinates is 2. -/
theorem plane_sphere_ratio_sum (a b c d e f p q r : ℝ) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (hdist : d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0)
  (hplane : a / d + b / e + c / f = 1)
  (hsphere : p^2 + q^2 + r^2 = (p - d)^2 + q^2 + r^2 ∧ 
             p^2 + q^2 + r^2 = p^2 + (q - e)^2 + r^2 ∧ 
             p^2 + q^2 + r^2 = p^2 + q^2 + (r - f)^2) :
  a / p + b / q + c / r = 2 := by
sorry

end NUMINAMATH_CALUDE_plane_sphere_ratio_sum_l68_6882


namespace NUMINAMATH_CALUDE_quadratic_one_root_l68_6824

theorem quadratic_one_root (b c : ℝ) 
  (h1 : ∃! x : ℝ, x^2 + b*x + c = 0)
  (h2 : b = 2*c - 1) : 
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l68_6824


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l68_6830

theorem sum_of_roots_equals_seven : ∃ (r₁ r₂ : ℝ), 
  r₁^2 - 7*r₁ + 10 = 0 ∧ 
  r₂^2 - 7*r₂ + 10 = 0 ∧ 
  r₁ + r₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l68_6830


namespace NUMINAMATH_CALUDE_at_least_one_alarm_probability_l68_6812

theorem at_least_one_alarm_probability (p_A p_B : ℝ) 
  (h_A : 0 ≤ p_A ∧ p_A ≤ 1) 
  (h_B : 0 ≤ p_B ∧ p_B ≤ 1) : 
  1 - (1 - p_A) * (1 - p_B) = p_A + p_B - p_A * p_B :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_alarm_probability_l68_6812


namespace NUMINAMATH_CALUDE_recipe_calculation_l68_6828

/-- Represents the relationship between flour, cookies, and sugar -/
structure RecipeRelation where
  flour_to_cookies : ℝ → ℝ  -- Function from flour to cookies
  flour_to_sugar : ℝ → ℝ    -- Function from flour to sugar

/-- Given the recipe relationships, prove the number of cookies and amount of sugar for 4 cups of flour -/
theorem recipe_calculation (r : RecipeRelation) 
  (h1 : r.flour_to_cookies 3 = 24)  -- 24 cookies from 3 cups of flour
  (h2 : r.flour_to_sugar 3 = 1.5)   -- 1.5 cups of sugar for 3 cups of flour
  (h3 : ∀ x y, r.flour_to_cookies (x * y) = r.flour_to_cookies x * y)  -- Linear relationship for cookies
  (h4 : ∀ x y, r.flour_to_sugar (x * y) = r.flour_to_sugar x * y)      -- Linear relationship for sugar
  : r.flour_to_cookies 4 = 32 ∧ r.flour_to_sugar 4 = 2 := by
  sorry

#check recipe_calculation

end NUMINAMATH_CALUDE_recipe_calculation_l68_6828


namespace NUMINAMATH_CALUDE_equation_solution_l68_6886

theorem equation_solution (x : ℝ) (h : 5 / (4 + 1/x) = 1) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l68_6886


namespace NUMINAMATH_CALUDE_no_real_solutions_l68_6821

theorem no_real_solutions : ¬∃ (x : ℝ), x > 0 ∧ x^(1/4) = 15 / (8 - 2 * x^(1/4)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l68_6821


namespace NUMINAMATH_CALUDE_money_division_l68_6827

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 2400 →
  r - q = 3000 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l68_6827


namespace NUMINAMATH_CALUDE_investment_interest_rate_l68_6832

theorem investment_interest_rate 
  (initial_investment : ℝ)
  (first_duration : ℝ)
  (first_rate : ℝ)
  (second_duration : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_duration = 9/12)
  (h3 : first_rate = 0.09)
  (h4 : second_duration = 9/12)
  (h5 : final_value = 17218.50) :
  ∃ s : ℝ, 
    s = 0.10 ∧ 
    final_value = initial_investment * (1 + first_duration * first_rate) * (1 + second_duration * s) := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l68_6832


namespace NUMINAMATH_CALUDE_linear_equation_implies_a_equals_negative_two_l68_6876

theorem linear_equation_implies_a_equals_negative_two (a : ℝ) : 
  (∀ x, (a - 2) * x^(|a| - 1) - 2 = 0 → ∃ m k, (a - 2) * x^(|a| - 1) - 2 = m * x + k) → 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_a_equals_negative_two_l68_6876


namespace NUMINAMATH_CALUDE_average_height_of_four_l68_6853

/-- Given the heights of four people with specific relationships, prove their average height --/
theorem average_height_of_four (zara_height brixton_height zora_height itzayana_height : ℕ) : 
  zara_height = 64 →
  brixton_height = zara_height →
  zora_height = brixton_height - 8 →
  itzayana_height = zora_height + 4 →
  (zara_height + brixton_height + zora_height + itzayana_height) / 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_average_height_of_four_l68_6853


namespace NUMINAMATH_CALUDE_quadratic_factorization_l68_6835

theorem quadratic_factorization (p q : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q)) →
  p - q = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l68_6835


namespace NUMINAMATH_CALUDE_arcade_candy_cost_l68_6829

theorem arcade_candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 33)
  (h2 : tickets_game2 = 9)
  (h3 : candies = 7) :
  (tickets_game1 + tickets_game2) / candies = 6 :=
by sorry

end NUMINAMATH_CALUDE_arcade_candy_cost_l68_6829


namespace NUMINAMATH_CALUDE_smallest_special_is_correct_l68_6898

/-- A natural number is special if it uses exactly four different digits in its decimal representation -/
def is_special (n : ℕ) : Prop :=
  (n.digits 10).toFinset.card = 4

/-- The smallest special number greater than 3429 -/
def smallest_special : ℕ := 3450

theorem smallest_special_is_correct :
  is_special smallest_special ∧
  smallest_special > 3429 ∧
  ∀ m : ℕ, m > 3429 → is_special m → m ≥ smallest_special :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_is_correct_l68_6898


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_l68_6807

/-- Represents an n-digit number with all digits equal to d -/
def digits_number (n : ℕ+) (d : ℕ) : ℕ :=
  d * (10^n.val - 1) / 9

/-- The equation C_n - A_n = B_n^2 holds for at least two distinct values of n -/
def equation_holds (a b c : ℕ) : Prop :=
  ∃ n m : ℕ+, n ≠ m ∧
    digits_number (2*n) c - digits_number n a = (digits_number n b)^2 ∧
    digits_number (2*m) c - digits_number m a = (digits_number m b)^2

theorem smallest_sum_of_digits :
  ∀ a b c : ℕ,
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    equation_holds a b c →
    ∀ x y z : ℕ,
      0 < x ∧ x < 10 →
      0 < y ∧ y < 10 →
      0 < z ∧ z < 10 →
      equation_holds x y z →
      5 ≤ x + y + z :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_l68_6807


namespace NUMINAMATH_CALUDE_always_unaffected_square_l68_6885

/-- Represents a square cut on the cake -/
structure Cut where
  x : ℚ
  y : ℚ
  size : ℚ
  h_x : 0 ≤ x ∧ x + size ≤ 3
  h_y : 0 ≤ y ∧ y + size ≤ 3

/-- Represents a small 1/3 x 1/3 square on the cake -/
structure SmallSquare where
  x : ℚ
  y : ℚ
  h_x : x = 0 ∨ x = 1 ∨ x = 2
  h_y : y = 0 ∨ y = 1 ∨ y = 2

/-- Check if a small square is affected by a cut -/
def isAffected (s : SmallSquare) (c : Cut) : Prop :=
  (c.x < s.x + 1/3 ∧ s.x < c.x + c.size) ∧
  (c.y < s.y + 1/3 ∧ s.y < c.y + c.size)

/-- Main theorem: There always exists an unaffected 1/3 x 1/3 square -/
theorem always_unaffected_square (cuts : Finset Cut) (h : cuts.card = 4) (h_size : ∀ c ∈ cuts, c.size = 1) :
  ∃ s : SmallSquare, ∀ c ∈ cuts, ¬isAffected s c :=
sorry

end NUMINAMATH_CALUDE_always_unaffected_square_l68_6885


namespace NUMINAMATH_CALUDE_hyperbola_equation_l68_6847

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 ∧ b > 0 →
  (∀ (x y : ℝ), y^2 = -8*x → (x + 2)^2 + y^2 = 4) →  -- Focus of parabola is (-2, 0)
  (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1 →  -- P lies on the hyperbola
  P = (2 * Real.sqrt 3, 2) →
  (∀ (x y : ℝ), x^2 / 4 - y^2 / 2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l68_6847


namespace NUMINAMATH_CALUDE_bananas_left_l68_6814

def dozen : Nat := 12

theorem bananas_left (initial : Nat) (eaten : Nat) : 
  initial = dozen → eaten = 1 → initial - eaten = 11 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l68_6814


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l68_6818

/-- The equation of the line passing through the center of the circle x^2 + 2x + y^2 = 0
    and perpendicular to the line x + y = 0 is x - y + 1 = 0. -/
theorem perpendicular_line_equation : ∃ (a b c : ℝ),
  (∀ x y : ℝ, x^2 + 2*x + y^2 = 0 → (x + 1)^2 + y^2 = 1) ∧ 
  (a*1 + b*1 = 0) ∧
  (a*x + b*y + c = 0 ↔ x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l68_6818


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l68_6867

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_transitive 
  (a b c : Line) :
  perpendicular a b → parallel_line b c → perpendicular a c :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l68_6867


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l68_6878

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l68_6878


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l68_6803

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l68_6803


namespace NUMINAMATH_CALUDE_weeks_to_save_for_shirt_l68_6864

/-- Calculate the number of weeks needed to save for a shirt -/
theorem weeks_to_save_for_shirt (total_cost saved_amount savings_rate : ℚ) : 
  total_cost = 3 →
  saved_amount = 3/2 →
  savings_rate = 1/2 →
  (total_cost - saved_amount) / savings_rate = 3 := by
  sorry

#check weeks_to_save_for_shirt

end NUMINAMATH_CALUDE_weeks_to_save_for_shirt_l68_6864


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l68_6894

theorem reciprocal_of_negative_fraction :
  ((-1 : ℚ) / 2011)⁻¹ = -2011 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l68_6894


namespace NUMINAMATH_CALUDE_mustard_bottles_sum_l68_6855

theorem mustard_bottles_sum : 0.25 + 0.25 + 0.38 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_mustard_bottles_sum_l68_6855


namespace NUMINAMATH_CALUDE_unique_solution_condition_l68_6859

theorem unique_solution_condition (t : ℝ) :
  (∃! x y z v : ℝ, x + y + z + v = 0 ∧ (x*y + y*z + z*v) + t*(x*z + x*v + y*v) = 0) ↔
  (t > (3 - Real.sqrt 5) / 2 ∧ t < (3 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l68_6859


namespace NUMINAMATH_CALUDE_senior_ticket_price_l68_6887

/-- Represents the cost of movie tickets for a family --/
structure MovieTickets where
  adult_price : ℕ
  child_price : ℕ
  total_cost : ℕ
  num_adults : ℕ
  num_children : ℕ
  num_seniors : ℕ

/-- Theorem stating the price of a senior citizen's ticket --/
theorem senior_ticket_price (tickets : MovieTickets) 
  (h1 : tickets.adult_price = 11)
  (h2 : tickets.child_price = 8)
  (h3 : tickets.total_cost = 64)
  (h4 : tickets.num_adults = 3)
  (h5 : tickets.num_children = 2)
  (h6 : tickets.num_seniors = 2) :
  tickets.total_cost = 
    tickets.num_adults * tickets.adult_price + 
    tickets.num_children * tickets.child_price + 
    tickets.num_seniors * 13 :=
by sorry

end NUMINAMATH_CALUDE_senior_ticket_price_l68_6887


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l68_6873

/-- Given a parabola with equation y² = 8x, prove its focus coordinates and directrix equation -/
theorem parabola_focus_and_directrix :
  ∀ (x y : ℝ), y^2 = 8*x →
  (∃ (focus_x focus_y : ℝ), focus_x = 2 ∧ focus_y = 0) ∧
  (∃ (k : ℝ), k = -2 ∧ ∀ (x : ℝ), x = k → x ∈ {x | x = -2}) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l68_6873


namespace NUMINAMATH_CALUDE_test_questions_count_l68_6831

theorem test_questions_count : 
  ∀ (total : ℕ), 
    (total % 4 = 0) →  -- The test has 4 sections with equal number of questions
    (20 : ℚ) / total > (60 : ℚ) / 100 → -- Correct answer percentage > 60%
    (20 : ℚ) / total < (70 : ℚ) / 100 → -- Correct answer percentage < 70%
    total = 32 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l68_6831


namespace NUMINAMATH_CALUDE_profit_thirty_for_thirtyfive_l68_6850

/-- Calculates the profit percentage when selling a different number of articles than the cost price basis -/
def profit_percentage (sold : ℕ) (cost_basis : ℕ) : ℚ :=
  let profit := cost_basis - sold
  (profit / sold) * 100

/-- Theorem stating that selling 30 articles at the price of 35 articles' cost results in a profit of 1/6 * 100% -/
theorem profit_thirty_for_thirtyfive :
  profit_percentage 30 35 = 100 / 6 := by
  sorry

end NUMINAMATH_CALUDE_profit_thirty_for_thirtyfive_l68_6850


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l68_6806

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initial_size : Nat
  small_cube_size : Nat
  removed_center_cubes : Nat
  removed_per_small_cube : Nat

/-- Calculates the surface area of the modified cube structure -/
def surface_area (c : ModifiedCube) : Nat :=
  let remaining_small_cubes := c.initial_size^3 / c.small_cube_size^3 - c.removed_center_cubes
  let surface_per_small_cube := 6 * c.small_cube_size^2 + 12 -- Original surface + newly exposed
  remaining_small_cubes * surface_per_small_cube

/-- Theorem stating the surface area of the specific modified cube -/
theorem modified_cube_surface_area :
  let c : ModifiedCube := {
    initial_size := 12,
    small_cube_size := 3,
    removed_center_cubes := 7,
    removed_per_small_cube := 9
  }
  surface_area c = 3762 := by sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l68_6806


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l68_6819

theorem smallest_integer_with_given_remainders : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (∀ y : ℕ, y > 0 → y % 3 = 2 → y % 4 = 3 → y % 5 = 4 → x ≤ y) ∧
  (x = 59) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l68_6819


namespace NUMINAMATH_CALUDE_stating_sock_drawing_probability_l68_6805

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 5

/-- 
Theorem stating the probability of drawing 5 socks with exactly one pair 
of the same color and the rest different colors, given 10 socks with 
2 socks each of 5 colors.
-/
theorem sock_drawing_probability : 
  (total_socks = 10) → 
  (num_colors = 5) → 
  (socks_per_color = 2) → 
  (socks_drawn = 5) →
  (Prob_exactly_one_pair_rest_different : ℚ) →
  Prob_exactly_one_pair_rest_different = 10 / 63 := by
  sorry

end NUMINAMATH_CALUDE_stating_sock_drawing_probability_l68_6805


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l68_6817

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≤ 1) ↔ (∃ x₀ : ℝ, x₀^2 > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l68_6817


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a1_l68_6893

theorem geometric_sequence_min_a1 (a : ℕ+ → ℕ+) (r : ℕ+) :
  (∀ i : ℕ+, a (i + 1) = a i * r) →  -- Geometric sequence condition
  (a 20 + a 21 = 20^21) →            -- Given condition
  (∃ x y : ℕ+, (∀ k : ℕ+, a 1 ≤ 2^(x:ℕ) * 5^(y:ℕ)) ∧ 
               a 1 = 2^(x:ℕ) * 5^(y:ℕ) ∧ 
               x + y = 24) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a1_l68_6893


namespace NUMINAMATH_CALUDE_smallest_prime_scalene_perimeter_l68_6874

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if three numbers form a scalene triangle -/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a scalene triangle with prime side lengths -/
theorem smallest_prime_scalene_perimeter :
  ∀ a b c : ℕ,
    isPrime a → isPrime b → isPrime c →
    isScaleneTriangle a b c →
    a + b + c ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_scalene_perimeter_l68_6874


namespace NUMINAMATH_CALUDE_unique_a_value_l68_6820

def A (a : ℝ) : Set ℝ := {0, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem unique_a_value : ∃! a : ℝ, A a ∪ B a = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l68_6820


namespace NUMINAMATH_CALUDE_function_extrema_sum_l68_6848

/-- Given f(x) = 2x^3 - ax^2 + 1 where a > 0, if the sum of the maximum and minimum values 
    of f(x) on [-1, 1] is 1, then a = 1/2 -/
theorem function_extrema_sum (a : ℝ) (h1 : a > 0) : 
  let f := fun x => 2 * x^3 - a * x^2 + 1
  (∃ M m : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ M ∧ m ≤ f x) ∧ M + m = 1) → 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l68_6848


namespace NUMINAMATH_CALUDE_solution_of_equation_l68_6826

theorem solution_of_equation (x : ℝ) : 2 * x - 4 * x = 0 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l68_6826


namespace NUMINAMATH_CALUDE_variations_difference_l68_6879

theorem variations_difference (n : ℕ) : n ^ 3 = n * (n - 1) * (n - 2) + 225 ↔ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_variations_difference_l68_6879


namespace NUMINAMATH_CALUDE_seventh_group_sample_l68_6896

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (populationSize : ℕ) (groupCount : ℕ) (firstNumber : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := populationSize / groupCount
  let lastTwoDigits := (firstNumber + 33 * groupNumber) % 100
  (groupNumber - 1) * interval + lastTwoDigits

/-- Theorem stating the result of the systematic sampling for the 7th group -/
theorem seventh_group_sample :
  systematicSample 1000 10 57 7 = 688 := by
  sorry

#eval systematicSample 1000 10 57 7

end NUMINAMATH_CALUDE_seventh_group_sample_l68_6896


namespace NUMINAMATH_CALUDE_tourists_travelers_checks_l68_6869

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : Nat
  hundred : Nat

/-- The problem statement -/
theorem tourists_travelers_checks 
  (tc : TravelersChecks)
  (h1 : 50 * tc.fifty + 100 * tc.hundred = 1800)
  (h2 : tc.fifty ≥ 24)
  (h3 : (1800 - 50 * 24) / (tc.fifty + tc.hundred - 24) = 100) :
  tc.fifty + tc.hundred = 30 := by
  sorry

end NUMINAMATH_CALUDE_tourists_travelers_checks_l68_6869


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l68_6877

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (l : Line) (α β : Plane) (h1 : α ≠ β) :
  perpendicular l α → perpendicular l β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l68_6877
