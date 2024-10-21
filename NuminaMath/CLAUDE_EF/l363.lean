import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_in_triangle_l363_36315

/-- Given a function f and a triangle ABC, prove the range of f(A) --/
theorem function_range_in_triangle (M ω φ : ℝ) (A B C : ℝ) (a b c : ℝ) :
  M > 0 →
  |φ| < π / 2 →
  0 < ω →
  ω < 3 →
  M * Real.sin (ω * (2 * π / 3) + φ) = 2 →
  M * Real.sin φ = 1 →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  A + B + C = π →
  A > 0 →
  A < 2 * π / 3 →
  let f := fun x ↦ M * Real.sin (ω * x + φ)
  ∀ x ∈ Set.Ioo 1 2, ∃ y ∈ Set.Ioo 0 (2 * π / 3), f y = x ∧
  ∀ y ∈ Set.Ioo 0 (2 * π / 3), f y ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_in_triangle_l363_36315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l363_36353

theorem salary_calculation (salary : ℝ) 
  (h1 : 17000 = salary - (salary * (1/5 : ℝ) + salary * (1/10 : ℝ) + salary * (3/5 : ℝ))) : 
  salary = 170000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l363_36353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l363_36321

/-- Calculates the loss percentage given the cost price and selling price -/
noncomputable def loss_percentage (cost_price selling_price : ℝ) : ℝ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem stating that the loss percentage for a radio with cost price 1800 and selling price 1430 is approximately 20.56% -/
theorem radio_loss_percentage :
  let cost_price : ℝ := 1800
  let selling_price : ℝ := 1430
  abs (loss_percentage cost_price selling_price - 20.56) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l363_36321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_after_breaks_l363_36398

/-- Represents a tripod with potentially broken legs -/
structure Tripod where
  original_leg_length : ℝ
  original_height : ℝ
  leg1_length : ℝ
  leg2_length : ℝ
  leg3_length : ℝ

/-- Calculates the new height of a tripod with broken legs -/
noncomputable def new_height (t : Tripod) : ℝ :=
  (t.original_height / t.original_leg_length) * ((t.leg1_length + t.leg2_length + t.leg3_length) / 3)

theorem tripod_height_after_breaks (t : Tripod) 
  (h_original_leg : t.original_leg_length = 6)
  (h_original_height : t.original_height = 5)
  (h_leg1 : t.leg1_length = 6)
  (h_leg2 : t.leg2_length = 5)
  (h_leg3 : t.leg3_length = 4) :
  new_height t = 25 / 6 := by
  -- Expand the definition of new_height
  unfold new_height
  -- Substitute known values
  rw [h_original_leg, h_original_height, h_leg1, h_leg2, h_leg3]
  -- Simplify the arithmetic
  simp [mul_div_assoc, add_div]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_after_breaks_l363_36398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l363_36395

/-- A dog is tied to a square shed with a rope. This function calculates the roaming area. -/
noncomputable def roamingArea (shedSide : ℝ) (ropeLength : ℝ) (distanceFromCorner : ℝ) : ℝ :=
  if distanceFromCorner = shedSide / 2 then
    -- Arrangement I: Semicircle
    (1/2) * Real.pi * ropeLength^2
  else
    -- Arrangement II: Semicircle + quarter circle
    (1/2) * Real.pi * ropeLength^2 + (1/4) * Real.pi * distanceFromCorner^2

/-- The theorem states that the difference in roaming area between Arrangement II and I is 4π sq ft. -/
theorem roaming_area_difference (shedSide : ℝ) (ropeLength : ℝ) :
  shedSide = 16 →
  ropeLength = 8 →
  roamingArea shedSide ropeLength 4 - roamingArea shedSide ropeLength (shedSide / 2) = 4 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l363_36395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_journey_third_part_distance_l363_36364

/-- Represents Toby's journey -/
structure TobyJourney where
  unloaded_speed : ℚ
  loaded_speed : ℚ
  part1_distance : ℚ
  part2_distance : ℚ
  part4_distance : ℚ
  total_time : ℚ

/-- Calculates the distance of the third part of Toby's journey -/
def third_part_distance (j : TobyJourney) : ℚ :=
  let part1_time := j.part1_distance / j.loaded_speed
  let part2_time := j.part2_distance / j.unloaded_speed
  let part4_time := j.part4_distance / j.unloaded_speed
  let part3_time := j.total_time - (part1_time + part2_time + part4_time)
  part3_time * j.loaded_speed

/-- Theorem stating that the third part distance is 80 miles -/
theorem toby_journey_third_part_distance :
  third_part_distance {
    unloaded_speed := 20,
    loaded_speed := 10,
    part1_distance := 180,
    part2_distance := 120,
    part4_distance := 140,
    total_time := 39
  } = 80 := by
  rw [third_part_distance]
  simp
  norm_num
  -- The proof is completed by normalization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_journey_third_part_distance_l363_36364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_average_age_l363_36371

theorem original_average_age (original_count new_count new_avg_age avg_decrease : ℕ) : ℕ := by
  let original_avg_age : ℕ := 40
  have h1 : original_count = 15 := by sorry
  have h2 : new_count = 15 := by sorry
  have h3 : new_avg_age = 32 := by sorry
  have h4 : avg_decrease = 4 := by sorry
  have h5 : (original_count * original_avg_age + new_count * new_avg_age) / (original_count + new_count) = original_avg_age - avg_decrease := by sorry
  exact original_avg_age

#check original_average_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_average_age_l363_36371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l363_36330

noncomputable def floor (x : ℝ) := ⌊x⌋
noncomputable def frac (x : ℝ) := x - ⌊x⌋

theorem system_solution (x y : ℝ) 
  (eq1 : floor x + frac y = 3.2)
  (eq2 : frac x + floor y = 6.3) :
  2 * x + 3 * y = 25.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l363_36330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrea_stop_HH_prob_l363_36324

/-- A coin flip can result in either heads or tails. -/
inductive CoinFlip
| Heads
| Tails

/-- The result of Andrea's coin flipping game. -/
inductive FlipResult
| StoppedHH  -- Stopped after flipping HH
| StoppedTH  -- Stopped after flipping TH

/-- Andrea's coin flipping game. -/
def andreaGame : Type :=
  List CoinFlip → Option FlipResult

/-- A fair coin has equal probability of heads and tails. -/
noncomputable def fairCoin : CoinFlip → ℝ
| CoinFlip.Heads => 1/2
| CoinFlip.Tails => 1/2

/-- The probability of Andrea stopping after flipping HH. -/
noncomputable def probStopHH : ℝ := 1/4

/-- Theorem: The probability of Andrea stopping after flipping HH is 1/4. -/
theorem andrea_stop_HH_prob :
  probStopHH = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrea_stop_HH_prob_l363_36324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l363_36361

/-- The time (in days) it takes for worker B to complete the job alone -/
noncomputable def B_time : ℚ := 27

/-- The time (in days) it takes for workers A and B to complete the job together -/
noncomputable def AB_time : ℚ := 18

/-- The efficiency ratio of worker A compared to worker B -/
noncomputable def A_efficiency_ratio : ℚ := 1/2

theorem job_completion_time :
  (1 / B_time + A_efficiency_ratio / B_time) * AB_time = 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l363_36361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_division_problem_l363_36306

theorem number_division_problem (x y m : ℝ) : 
  x + y = 52 →
  m * x + 22 * y = 780 →
  y = 30.333333333333332 →
  abs (m - 5.1) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_division_problem_l363_36306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_distances_l363_36313

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a lighthouse -/
structure Lighthouse where
  position : Point

/-- Represents an airplane -/
structure Airplane where
  initialPosition : Point
  speed : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def angleBetween (p1 p2 p3 : Point) : ℝ :=
  Real.arccos ((distance p1 p2)^2 + (distance p1 p3)^2 - (distance p2 p3)^2) /
               (2 * distance p1 p2 * distance p1 p3)

theorem lighthouse_distances (plane : Airplane) (l1 l2 : Lighthouse) :
  plane.speed = 432 ∧
  angleBetween plane.initialPosition l1.position (Point.mk 0 1) = π/4 ∧
  angleBetween plane.initialPosition l2.position (Point.mk 0 1) = π/8 ∧
  angleBetween (Point.mk plane.initialPosition.x (plane.initialPosition.y + plane.speed * (5/60))) l1.position (Point.mk 0 1) = 3*π/8 ∧
  angleBetween (Point.mk plane.initialPosition.x (plane.initialPosition.y + plane.speed * (5/60))) l2.position (Point.mk 0 1) = π/4 →
  ∃ (d1 d2 : ℝ), abs (d1 - 86.9) < 0.1 ∧ abs (d2 - 66.6) < 0.1 ∧
    d1 = distance (Point.mk plane.initialPosition.x (plane.initialPosition.y + plane.speed * (5/60))) l1.position ∧
    d2 = distance (Point.mk plane.initialPosition.x (plane.initialPosition.y + plane.speed * (5/60))) l2.position :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_distances_l363_36313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hurricane_average_rainfall_l363_36326

/-- Represents the rainfall data for a storm --/
structure StormData where
  initial_rainfall : ℝ
  initial_duration : ℝ
  secondary_rainfall : ℝ
  secondary_duration : ℝ

/-- Calculates the total rainfall for a storm --/
def total_rainfall (storm : StormData) : ℝ :=
  storm.initial_rainfall + storm.secondary_rainfall

/-- Calculates the total duration for a storm --/
def total_duration (storm : StormData) : ℝ :=
  storm.initial_duration + storm.secondary_duration

/-- Theorem: The overall average rainfall amount for the hurricane system is approximately 2.87 inches per hour --/
theorem hurricane_average_rainfall 
  (storm_a : StormData)
  (storm_b : StormData)
  (storm_c : StormData)
  (h_a : storm_a = { initial_rainfall := 5, initial_duration := 0.5, 
                     secondary_rainfall := 2.5, secondary_duration := 0.5 })
  (h_b : storm_b = { initial_rainfall := 3, initial_duration := 0.75, 
                     secondary_rainfall := 4.5, secondary_duration := 1 })
  (h_c : storm_c = { initial_rainfall := 1.5, initial_duration := 3, 
                     secondary_rainfall := 0, secondary_duration := 0 }) :
  let total_rain := total_rainfall storm_a + total_rainfall storm_b + total_rainfall storm_c
  let total_time := total_duration storm_a + total_duration storm_b + total_duration storm_c
  |((total_rain / total_time) - 2.87)| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hurricane_average_rainfall_l363_36326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_decreases_l363_36396

/-- Atomic mass of Carbon -/
noncomputable def mass_C : ℝ := 12.01

/-- Atomic mass of Hydrogen -/
noncomputable def mass_H : ℝ := 1.008

/-- Atomic mass of Oxygen -/
noncomputable def mass_O : ℝ := 16.00

/-- Molar mass of C6H8Ox -/
noncomputable def molar_mass (x : ℕ) : ℝ := 6 * mass_C + 8 * mass_H + x * mass_O

/-- Mass of Carbon in C6H8Ox -/
noncomputable def mass_carbon : ℝ := 6 * mass_C

/-- Mass percentage of Carbon in C6H8Ox -/
noncomputable def mass_percentage_C (x : ℕ) : ℝ := (mass_carbon / molar_mass x) * 100

/-- Given: Mass percentage of Carbon in C6H8O6 is approximately 40.91% -/
axiom mass_percentage_C_6 : ∃ ε > 0, |mass_percentage_C 6 - 40.91| < ε

/-- Theorem: As x increases, the mass percentage of Carbon in C6H8Ox decreases -/
theorem mass_percentage_C_decreases (x y : ℕ) (h : x < y) :
  mass_percentage_C y < mass_percentage_C x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_C_decreases_l363_36396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_exists_l363_36351

-- Define the board as a 3x3 matrix of natural numbers
def Board := Matrix (Fin 3) (Fin 3) ℕ

-- Define a type for the 9 distinct cards
def Cards := { cards : Vector ℕ 9 // cards.toList.Nodup }

-- Define the win condition for the first player
def FirstPlayerWins (board : Board) : Prop :=
  board 0 0 + board 0 1 + board 0 2 + board 2 0 + board 2 1 + board 2 2 >
  board 0 0 + board 1 0 + board 2 0 + board 0 2 + board 1 2 + board 2 2

-- Define a strategy as a function that takes the current board state and available cards
-- and returns the position to place the next card
def Strategy := Board → Cards → Fin 3 × Fin 3

-- Define a function to simulate the game and return the final board
def playGame (s : Strategy) (cards : Cards) : Board :=
  sorry -- Implementation of game simulation

-- Theorem statement
theorem first_player_winning_strategy_exists :
  ∃ (s : Strategy), ∀ (cards : Cards),
    FirstPlayerWins (playGame s cards) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_exists_l363_36351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_solution_set_l363_36392

noncomputable def f (x : ℝ) : ℝ := |x - Real.sqrt 2| + |x + 2 * Real.sqrt 2|

theorem f_max_and_solution_set :
  (∃ (M : ℝ), M = (5/2) * Real.sqrt 2 ∧ 
    (∀ (x : ℝ), f x ≤ M) ∧
    (∃ (x : ℝ), f x = M)) ∧
  {x : ℝ | f x ≤ (5/2) * Real.sqrt 2} = {x : ℝ | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_solution_set_l363_36392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_curve_area_l363_36385

/-- The area of the region bounded by the curve traced by the centroid of a triangle inscribed in a circle -/
theorem centroid_curve_area (r : ℝ) (h : r > 0) : 
  (2 * r = 30) → 
  (25 : ℝ) * Real.pi = Real.pi * ((r / 3) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_curve_area_l363_36385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l363_36309

theorem expansion_coefficient (n : ℕ) : 
  (∀ k, k ≠ 7 → (n.choose k) ≤ (n.choose 7)) → 
  (n.choose 7) > (n.choose 6) →
  (n.choose 7) > (n.choose 8) →
  (∃ r : ℕ, ((-1)^r : ℤ) * (n.choose r) = -792 ∧ 2*r - n = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l363_36309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l363_36348

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_properties :
  /- f is defined on (-1,1) -/
  (∀ x, -1 < x ∧ x < 1 → f x ≠ 0) ∧
  /- f is an odd function -/
  (∀ x, -1 < x ∧ x < 1 → f (-x) = -(f x)) ∧
  /- f is monotonically increasing on (-1,1) -/
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l363_36348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_count_l363_36314

/-- A point in the triangular lattice -/
structure LatticePoint where
  x : ℚ
  y : ℚ

/-- The triangular lattice -/
def TriangularLattice : Set LatticePoint :=
  sorry

/-- Distance between two lattice points -/
def distance (p q : LatticePoint) : ℚ :=
  sorry

/-- An equilateral triangle in the lattice -/
structure EquilateralTriangle where
  p1 : LatticePoint
  p2 : LatticePoint
  p3 : LatticePoint
  h1 : p1 ∈ TriangularLattice
  h2 : p2 ∈ TriangularLattice
  h3 : p3 ∈ TriangularLattice
  h4 : distance p1 p2 = distance p2 p3
  h5 : distance p2 p3 = distance p3 p1

/-- The set of all equilateral triangles in the lattice -/
def AllEquilateralTriangles : Finset EquilateralTriangle :=
  sorry

/-- The count of equilateral triangles in the lattice -/
theorem equilateral_triangles_count :
  Finset.card AllEquilateralTriangles = 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_count_l363_36314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l363_36325

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 3 (1/10))
  (hb : b = Real.rpow (1/3) (-4/5))
  (hc : c = Real.log 0.8 / Real.log 0.7) : 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l363_36325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_grocery_bill_l363_36387

noncomputable def banana_price : ℝ := 2
noncomputable def bread_price : ℝ := 3
noncomputable def butter_price : ℝ := 5
noncomputable def cereal_price : ℝ := 6

def banana_quantity : ℕ := 6
def bread_quantity : ℕ := 2
def butter_quantity : ℕ := 3
def cereal_quantity : ℕ := 4

noncomputable def cereal_discount : ℝ := 0.25
noncomputable def coupon_threshold : ℝ := 50
noncomputable def coupon_value : ℝ := 10

noncomputable def grocery_total : ℝ :=
  banana_price * (banana_quantity : ℝ) +
  bread_price * (bread_quantity : ℝ) +
  butter_price * (butter_quantity : ℝ) +
  cereal_price * (1 - cereal_discount) * (cereal_quantity : ℝ)

noncomputable def final_payment : ℝ :=
  if grocery_total ≥ coupon_threshold then
    grocery_total - coupon_value
  else
    grocery_total

theorem johns_grocery_bill : final_payment = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_grocery_bill_l363_36387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_combined_list_l363_36310

def list_length : ℕ := 6060

def first_part (n : ℕ) : List ℕ := List.range n

def second_part (n : ℕ) : List ℕ := List.map (λ x => x * x) (List.range n)

def combined_list (n : ℕ) : List ℕ := (first_part n) ++ (second_part n)

theorem median_of_combined_list :
  let n := 3030
  let list := combined_list n
  list.length = list_length ∧ 
  (∃ a b : ℕ, a ∈ list ∧ b ∈ list ∧ (a + b : ℚ) / 2 = 2975.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_combined_list_l363_36310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_and_quadratic_roots_l363_36334

-- Define the determinant inequality
def det_inequality (m : ℝ) : Set ℝ :=
  {x | (x + m) * x - 2 < 0}

-- Define the quadratic equation
def quadratic_equation (n : ℝ) : ℂ → Prop :=
  λ x ↦ x^2 - x + n = 0

theorem determinant_and_quadratic_roots :
  (∃ m : ℝ, det_inequality m = Set.Ioo (-1) 2) ∧
  (∃ n : ℝ, quadratic_equation n (Complex.ofReal (1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2))) :=
by
  constructor
  · use -1
    sorry
  · use 1
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_and_quadratic_roots_l363_36334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_monthly_rent_l363_36312

/-- The monthly rent required for a property investment --/
noncomputable def monthly_rent (investment : ℝ) (return_rate : ℝ) (annual_taxes : ℝ) (annual_insurance : ℝ) (maintenance_rate : ℝ) : ℝ :=
  let annual_expenses := return_rate * investment + annual_taxes + annual_insurance
  annual_expenses / (12 * (1 - maintenance_rate))

/-- Theorem stating the correct monthly rent for the given problem --/
theorem correct_monthly_rent :
  let R := monthly_rent 12000 0.06 360 240 0.1
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |R - 122.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_monthly_rent_l363_36312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_amount_l363_36393

/-- Represents the number of pennies in the bag -/
def x : ℕ → ℕ := id

/-- The total value of coins in the bag in cents -/
def total_value (n : ℕ) : ℕ := n + 20*n + 150*n

/-- The possible amounts in the bag in cents -/
def possible_amounts : List ℕ := [30600, 33300, 34200, 34800, 36000]

/-- Theorem stating that 34200 (corresponding to $342) is the only amount in the list that can be expressed as the total value for some number of pennies -/
theorem correct_amount : 
  ∃ (n : ℕ), total_value n = 34200 ∧ 
  ∀ (m : ℕ), m ∈ possible_amounts ∧ m ≠ 34200 → ¬∃ (k : ℕ), total_value k = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_amount_l363_36393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l363_36362

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (ρ₁ : ℝ) (θ₁ : ℝ) (ρ₂ : ℝ) (θ₂ : ℝ) : ℝ :=
  Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2)

theorem distance_between_polar_points :
  polar_distance 2 (π/6) 4 (5*π/6) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l363_36362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l363_36335

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 3)) / x

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -3 ∧ x ≠ 0}

-- Theorem stating that domain_f is indeed the domain of f
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l363_36335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercepts_l363_36346

/-- An ellipse with foci at (0, 3) and (4, 0) and one x-intercept at the origin has its other two x-intercepts at (-3/2, 0) and (11/2, 0). -/
theorem ellipse_x_intercepts :
  ∀ (E : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ E ↔ Real.sqrt ((x - 0)^2 + (y - 3)^2) + Real.sqrt ((x - 4)^2 + y^2) = 7) →
  (0, 0) ∈ E →
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ (x₁, 0) ∈ E ∧ (x₂, 0) ∈ E ∧ x₁ = -3/2 ∧ x₂ = 11/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercepts_l363_36346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_readers_difference_l363_36316

theorem book_readers_difference (total : ℕ) 
  (a_and_b_ratio : ℚ) (b_and_a_ratio : ℚ) :
  total = 800 →
  a_and_b_ratio = 1/5 →
  b_and_a_ratio = 1/4 →
  ∃ (only_a only_b both : ℕ),
    only_a + only_b + both = total ∧
    (both : ℚ) = a_and_b_ratio * (only_a + both : ℚ) ∧
    (both : ℚ) = b_and_a_ratio * (only_b + both : ℚ) ∧
    only_a - only_b = 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_readers_difference_l363_36316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_l363_36355

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt (1 - x^2) - Real.sin x

-- State the theorem
theorem integral_equals_pi :
  ∫ x in Set.Icc (-1) 1, f x = π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_l363_36355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_spent_l363_36366

/-- Calculates the total amount spent on pens and pencils given the quantities and prices. -/
theorem total_amount_spent 
  (num_pens : ℕ) 
  (num_pencils : ℕ) 
  (price_pen : ℚ) 
  (price_pencil : ℚ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : price_pen = 10)
  (h4 : price_pencil = 2) :
  (num_pens : ℚ) * price_pen + (num_pencils : ℚ) * price_pencil = 450 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_spent_l363_36366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_implies_a_l363_36357

open Real

/-- The function f(x) = ln x + a/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a / x

/-- The minimum value of f(x) on [1, e] is 3/2 -/
def min_value (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 (exp 1), f a x ≥ 3/2

/-- If the minimum value of f(x) on [1, e] is 3/2, then a = √e -/
theorem f_min_value_implies_a (a : ℝ) : min_value a → a = sqrt (exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_implies_a_l363_36357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_change_l363_36322

/-- Given a point M with coordinates (x, y) and a new origin O' with coordinates (a, b),
    the new coordinates of M with respect to O' are (x - a, y - b). -/
theorem coordinate_change (x y a b : ℝ) :
  let M : ℝ × ℝ := (x, y)
  let O' : ℝ × ℝ := (a, b)
  let new_coordinates : ℝ × ℝ := (x - a, y - b)
  new_coordinates = (x - a, y - b) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_change_l363_36322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l363_36376

open Real

/-- The function f(x) = ln(x) / x -/
noncomputable def f (x : ℝ) : ℝ := (log x) / x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := (1 - log x) / (x^2)

theorem tangent_parallel_to_x_axis (x₀ : ℝ) (h₁ : x₀ > 0) (h₂ : f_deriv x₀ = 0) :
  f x₀ = 1 / ℇ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l363_36376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_two_eq_281_div_16_l363_36333

noncomputable def t (x : ℝ) : ℝ := 4 * x - 9

noncomputable def s (y : ℝ) : ℝ := 
  let x := (y + 9) / 4
  x^2 + 4 * x - 1

theorem s_of_two_eq_281_div_16 : s 2 = 281 / 16 := by
  -- Unfold the definition of s
  unfold s
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_two_eq_281_div_16_l363_36333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l363_36372

/-- The distance between two points in a Cartesian coordinate system -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: The distance between points A(3,-2) and B(3,3) is 5 units -/
theorem distance_A_to_B : distance 3 (-2) 3 3 = 5 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l363_36372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l363_36304

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the constants a, b, and c
noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 25^(1/3)

-- State the theorem
theorem f_inequality (hf1 : ∀ x, f (-x) = f x) 
                     (hf2 : ∀ x, x < 0 → f x = 3^x + 1) : 
  f c < f a ∧ f a < f b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l363_36304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l363_36369

theorem ellipse_eccentricity (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, y^2 + m*x^2 = 1 → 
    (Real.sqrt (1 - min m (1/m)) = 1/2 ∨ Real.sqrt (1 - max m (1/m)) = 1/2)) → 
  (m = 3/4 ∨ m = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l363_36369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l363_36356

open Real Set

theorem periodic_function_property (f : ℝ → ℝ) (T : ℝ) (h_periodic : ∀ x, f (x + T) = f x) 
  (h_positive : T > 0) (h_continuous_deriv : Continuous (deriv f)) :
  ∃ x y, x ∈ Icc 0 T ∧ y ∈ Icc 0 T ∧ x ≠ y ∧ f x * (deriv f) y = (deriv f) x * f y := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l363_36356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susie_pizza_slices_l363_36339

/-- Susie's pizza store problem -/
theorem susie_pizza_slices : 
  ∃ (price_per_slice price_whole_pizza whole_pizzas_sold total_earnings : ℕ),
    price_per_slice = 3 ∧
    price_whole_pizza = 15 ∧
    whole_pizzas_sold = 3 ∧
    total_earnings = 117 ∧
    (price_per_slice * (total_earnings - price_whole_pizza * whole_pizzas_sold) / price_per_slice) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susie_pizza_slices_l363_36339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_monthly_installment_l363_36370

/-- Calculates the monthly installment for a car purchase on hire-purchase terms. -/
noncomputable def monthly_installment (cash_price : ℝ) (deposit_percentage : ℝ) (num_installments : ℕ) (annual_interest_rate : ℝ) : ℝ :=
  let deposit := deposit_percentage * cash_price
  let balance := cash_price - deposit
  let years := (num_installments : ℝ) / 12
  let total_amount := balance * (1 + annual_interest_rate * years)
  total_amount / (num_installments : ℝ)

/-- Theorem stating that the monthly installment for the given car purchase scenario is $504. -/
theorem car_monthly_installment :
  monthly_installment 21000 0.1 60 0.12 = 504 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_monthly_installment_l363_36370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_last_component_l363_36319

/-- Given two vectors a and b in ℝ³, if they are parallel and have the given components, then the last component of b must be -4. -/
theorem parallel_vectors_last_component (a b : ℝ × ℝ × ℝ) :
  a = (2, -1, 2) →
  b.1 = -4 →
  b.2.1 = 2 →
  (∃ (k : ℝ), a = k • b) →
  b.2.2 = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_last_component_l363_36319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheetah_reindeer_chase_l363_36303

/-- The time (in hours) it takes for the cheetah to catch up to the reindeer -/
noncomputable def catch_up_time : ℝ := 3 / 60

/-- The speed of the reindeer in miles per hour -/
noncomputable def reindeer_speed : ℝ := 50

/-- The speed of the cheetah in miles per hour -/
noncomputable def cheetah_speed : ℝ := 60

/-- The time difference (in hours) between when the reindeer and cheetah pass the tree -/
noncomputable def time_difference : ℝ := (cheetah_speed * catch_up_time - reindeer_speed * catch_up_time) / 
                           (cheetah_speed - reindeer_speed)

theorem cheetah_reindeer_chase :
  time_difference * 60 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheetah_reindeer_chase_l363_36303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l363_36365

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/2)

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → 
  2 * x₁^2 - 3 * x₁ + 1 > 0 → 2 * x₂^2 - 3 * x₂ + 1 > 0 →
  f x₂ < f x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l363_36365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sum_l363_36378

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else 9^(-x) + 1

theorem f_composition_sum : f (f 1) + f (-Real.log 2 / Real.log 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_sum_l363_36378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_symmetry_l363_36332

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem smallest_shift_for_symmetry 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi) = f ω x) 
  (m : ℝ) 
  (h_m_pos : m > 0) 
  (h_symmetry : ∀ x, f ω (x - m) = -f ω (-x - m)) :
  m = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_symmetry_l363_36332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carter_income_l363_36305

/-- Represents the tax structure and Carter's income --/
structure TaxSystem where
  q : ℝ  -- Base tax rate in decimal form
  income : ℝ  -- Carter's annual income

/-- Calculates the tax amount based on the given tax system --/
noncomputable def calculateTax (ts : TaxSystem) : ℝ :=
  let baseTax := ts.q * 30000
  let excessTax := (ts.q + 0.03) * (ts.income - 30000)
  baseTax + max excessTax 0

/-- Theorem stating that Carter's income is closest to $34000 --/
theorem carter_income (ts : TaxSystem) : 
  (calculateTax ts = (ts.q + 0.0035) * ts.income) → 
  (abs (ts.income - 34000) ≤ abs (ts.income - 30000) ∧
   abs (ts.income - 34000) ≤ abs (ts.income - 33000) ∧
   abs (ts.income - 34000) ≤ abs (ts.income - 35000) ∧
   abs (ts.income - 34000) ≤ abs (ts.income - 37000)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carter_income_l363_36305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_poles_approx_l363_36374

/-- The distance between fence poles for a rectangular plot -/
noncomputable def distance_between_poles (length width : ℝ) (num_poles : ℕ) : ℝ :=
  2 * (length + width) / (num_poles - 1 : ℝ)

/-- Theorem: The distance between poles for a 90m by 50m plot with 14 poles is approximately 21.54m -/
theorem distance_between_poles_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |distance_between_poles 90 50 14 - 21.54| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_poles_approx_l363_36374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_hit_origin_five_five_l363_36341

def move_probability : ℚ := 1 / 3

def is_on_axis (x y : ℕ) : Prop := x = 0 ∨ y = 0

def probability_hit_origin : ℕ → ℕ → ℚ
  | 0, 0 => 1
  | x, y =>
    if x = 0 ∨ y = 0 then 0
    else move_probability * (
      probability_hit_origin (x - 1) y +
      probability_hit_origin x (y - 1) +
      probability_hit_origin (x - 1) (y - 1)
    )

theorem probability_hit_origin_five_five :
  probability_hit_origin 5 5 = 1 / (3^5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_hit_origin_five_five_l363_36341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l363_36342

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x - 8)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x ≤ -2 ∨ x ≥ 4

-- Define the inner function u(x)
def u (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Theorem stating that the monotonic increasing interval of f(x) is [4, +∞)
theorem monotonic_increasing_interval :
  ∀ x y : ℝ, domain x → domain y → x ≥ 4 → y > x → f y > f x :=
by
  -- The proof is omitted and replaced with sorry
  sorry

-- Additional lemma to show that f is well-defined on its domain
lemma f_well_defined (x : ℝ) (h : domain x) : u x ≥ 0 :=
by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l363_36342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_first_to_last_l363_36375

/-- Represents the number of trees along the road -/
def num_trees : ℕ := 8

/-- Represents the distance between the first and fifth tree in feet -/
noncomputable def distance_1_to_5 : ℝ := 100

/-- Represents the distance between consecutive trees -/
noncomputable def distance_between_trees : ℝ := distance_1_to_5 / 4

/-- The theorem to be proved -/
theorem distance_first_to_last : 
  distance_between_trees * (num_trees - 1 : ℝ) = 175 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_first_to_last_l363_36375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l363_36377

/-- The distance from a point to a line --/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point (2, 1) to the line y = 1/2x + 1 is 2√5/5 --/
theorem distance_point_to_line_example : 
  distance_point_to_line 2 1 (1/2) (-1) 1 = 2 * Real.sqrt 5 / 5 := by
  sorry

#check distance_point_to_line_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l363_36377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_graph_l363_36317

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def areaEnclosedByGraph : ℝ := 96

/-- The equation of the graph -/
def graphEquation (x y : ℝ) : Prop := abs x + abs (3 * y) = 12

theorem area_enclosed_by_graph :
  areaEnclosedByGraph = 96 := by
  sorry

#eval areaEnclosedByGraph

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_graph_l363_36317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_calculation_l363_36397

/-- Calculate the total amount paid for fruits with discount and tax --/
theorem fruit_purchase_calculation (grapes_kg : ℝ) (grapes_price : ℝ) 
  (mangoes_kg : ℝ) (mangoes_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  grapes_kg = 8 ∧ grapes_price = 70 ∧ mangoes_kg = 9 ∧ mangoes_price = 65 ∧ 
  discount_rate = 0.1 ∧ tax_rate = 0.05 →
  (grapes_kg * grapes_price + mangoes_kg * mangoes_price) * (1 - discount_rate) * (1 + tax_rate) = 1082.025 := by
  sorry

#eval (8 * 70 + 9 * 65) * (1 - 0.1) * (1 + 0.05)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_calculation_l363_36397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l363_36394

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_inequality_solution_set (x : ℝ) : 
  (floor x)^2 - 5*(floor x) + 6 ≤ 0 ↔ x ∈ Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l363_36394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_smart_integers_div_by_18_l363_36329

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry  -- Implementation not provided, as it's not directly relevant to the proof

/-- Definition of a smart integer -/
def SmartInteger (n : ℕ) : Prop :=
  Even n ∧ 100 < n ∧ n < 300 ∧ (sumOfDigits n = 12)

/-- Theorem: All smart integers are divisible by 18 -/
theorem all_smart_integers_div_by_18 :
  ∀ n : ℕ, SmartInteger n → n % 18 = 0 :=
by
  sorry

#check all_smart_integers_div_by_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_smart_integers_div_by_18_l363_36329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l363_36340

open Real

-- Define the vectors m and n
noncomputable def m (α : ℝ) : ℝ × ℝ := (cos α - sqrt 2 / 3, -1)
noncomputable def n (α : ℝ) : ℝ × ℝ := (sin α, 1)

-- Define the collinearity condition
def collinear (α : ℝ) : Prop :=
  (m α).1 * (n α).2 = (m α).2 * (n α).1

-- Main theorem
theorem vector_problem (α : ℝ) (h1 : α ∈ Set.Icc (-π) 0) (h2 : collinear α) :
  (sin α + cos α = sqrt 2 / 3) ∧
  (sin (2 * α) / (sin α - cos α) = 7 / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l363_36340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_romano_cheese_cost_l363_36380

/-- The cost per kilogram of a special cheese blend -/
noncomputable def special_blend_cost : ℝ := 696.05

/-- The cost per kilogram of mozzarella cheese -/
noncomputable def mozzarella_cost : ℝ := 504.35

/-- The weight of mozzarella cheese in kilograms -/
noncomputable def mozzarella_weight : ℝ := 19

/-- The weight of romano cheese in kilograms -/
noncomputable def romano_weight : ℝ := 19

/-- The total weight of the cheese blend in kilograms -/
noncomputable def total_weight : ℝ := mozzarella_weight + romano_weight

/-- The cost per kilogram of romano cheese -/
noncomputable def romano_cost : ℝ := (special_blend_cost * total_weight - mozzarella_cost * mozzarella_weight) / romano_weight

theorem romano_cheese_cost : ∃ ε > 0, |romano_cost - 887.75| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_romano_cheese_cost_l363_36380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l363_36389

-- Define the triangles and their properties
structure Triangle where
  F : ℝ
  G : ℝ
  H : ℝ

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the theorem
theorem similar_triangles_side_length 
  (FGH KLM : Triangle) 
  (h_similar : similar FGH KLM) 
  (h_GH : FGH.H - FGH.G = 25) 
  (h_KM : KLM.H - KLM.F = 15) 
  (h_KL : KLM.G - KLM.F = 18) : 
  FGH.G - FGH.F = 30 := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l363_36389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_sum_powers_inequality_l363_36302

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Statement 1
theorem f_nonnegative_iff (a : ℝ) : 
  (∀ x ≥ 0, f a x ≥ 0) ↔ a ≤ 1 :=
sorry

-- Helper function for the sum in Statement 2
def sum_powers (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => (3 * i + 1) ^ n)

-- Statement 2
theorem sum_powers_inequality (n : ℕ) (hn : n ≥ 2) :
  (sum_powers n : ℝ) < (Real.exp (1/3) / (Real.exp 1 - 1)) * (3 * n : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_sum_powers_inequality_l363_36302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_eight_to_negative_five_thirds_l363_36363

theorem negative_eight_to_negative_five_thirds :
  (-8 : ℝ) ^ (-5/3 : ℝ) = -1/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_eight_to_negative_five_thirds_l363_36363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l363_36318

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.cos (2 * α) = -7/25) 
  (h2 : 0 < α) 
  (h3 : α < Real.pi/2) : 
  Real.cos α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l363_36318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_sufficient_not_necessary_l363_36367

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Slope of the first line ax+2y+3a=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a / 2

/-- Slope of the second line (a+1)x-3y+4=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := (a + 1) / 3

/-- The statement that a=2 is sufficient but not necessary for perpendicularity -/
theorem a_eq_2_sufficient_not_necessary :
  (perpendicular (slope1 2) (slope2 2)) ∧
  (∃ a : ℝ, a ≠ 2 ∧ perpendicular (slope1 a) (slope2 a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_sufficient_not_necessary_l363_36367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_error_is_0_16_percent_l363_36328

/-- Conversion factor from kilolunas to kilograms -/
noncomputable def kl_to_kg (mass : ℝ) : ℝ := (mass / 4) * 0.96

/-- Neznaika's method for converting kilograms to kilolunas -/
noncomputable def neznaika_kg_to_kl (mass : ℝ) : ℝ := (mass * 4) * 1.04

/-- The correct conversion from kilograms to kilolunas -/
noncomputable def kg_to_kl (mass : ℝ) : ℝ := mass / (kl_to_kg 1)

/-- The percentage error of Neznaika's method -/
noncomputable def neznaika_error : ℝ := 
  (1 - neznaika_kg_to_kl 1 / kg_to_kl 1) * 100

theorem neznaika_error_is_0_16_percent : 
  ∀ ε > 0, |neznaika_error - 0.16| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_neznaika_error_is_0_16_percent_l363_36328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_residue_solutions_l363_36350

/-- Legendre symbol -/
noncomputable def legendre (a p : ℤ) : ℤ := sorry

theorem quadratic_residue_solutions (m a : ℕ) :
  let p₁ := 4*m + 3
  let p₂ := 8*m + 5
  (Nat.Prime p₁ ∧ legendre (a : ℤ) p₁ = 1) →
    (∃ x₀ : ℤ, x₀ = (a : ℤ)^(m+1) ∨ x₀ = -(a : ℤ)^(m+1)) ∧ x₀^2 ≡ a [ZMOD p₁] ∧
  (Nat.Prime p₂ ∧ legendre (a : ℤ) p₂ = 1) →
    (∃ x₀ : ℤ, (x₀ = (a : ℤ)^(m+1) ∨ x₀ = -(a : ℤ)^(m+1) ∨ 
                x₀ = 2^(2*m+1)*(a : ℤ)^(m+1) ∨ x₀ = -(2^(2*m+1)*(a : ℤ)^(m+1)))) ∧ 
    x₀^2 ≡ a [ZMOD p₂] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_residue_solutions_l363_36350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unmarried_women_l363_36320

theorem max_unmarried_women (total : ℕ) (women_ratio : ℚ) (married_ratio : ℚ) 
  (h_total : total = 80)
  (h_women : women_ratio = 2 / 5)
  (h_married : married_ratio = 1 / 2) :
  (Nat.floor (women_ratio * total) : ℕ) = 32 ∧ 
  (Nat.floor (women_ratio * total) : ℕ) ≤ total - (Nat.floor (married_ratio * total) : ℕ) := by
  sorry

#check max_unmarried_women

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unmarried_women_l363_36320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l363_36349

noncomputable def is_solution (x : Real) : Prop :=
  0 < x ∧ x < 180 ∧ 
  (Real.cos (2 * x * Real.pi / 180))^3 + (Real.cos (4 * x * Real.pi / 180))^3 = 
  8 * (Real.cos (3 * x * Real.pi / 180))^3 * (Real.cos (x * Real.pi / 180))^3

theorem sum_of_solutions : 
  ∃ (S : Finset Real), (∀ x ∈ S, is_solution x) ∧ (S.sum id = 540) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l363_36349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_g100_l363_36300

def g (m : ℕ) : ℕ := (List.range (m - 1)).map (· + 2) |>.prod

theorem greatest_prime_factor_g100 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ g 100 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ g 100 → q ≤ p ∧ p = 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_g100_l363_36300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_marking_l363_36391

/-- A marking of cells on a grid. -/
def Marking (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle contains a marked cell. -/
def containsMarkedCell (n : ℕ) (m : Marking n) (x y w h : ℕ) : Prop :=
  ∃ (i : Fin w) (j : Fin h), 
    ∃ (xi : Fin n) (yj : Fin n), 
      (xi.val = x + i.val) ∧ (yj.val = y + j.val) ∧ m xi yj = true

/-- A valid marking satisfies the problem conditions. -/
def validMarking (n : ℕ) (m : Marking n) : Prop :=
  (∀ x y w h, w * h ≥ n → x + w ≤ n → y + h ≤ n → containsMarkedCell n m x y w h) ∧
  (∃ (S : Finset (Fin n × Fin n)), S.card = n ∧ ∀ (i j : Fin n), m i j = (⟨i, j⟩ ∈ S))

/-- The main theorem: the largest n for a valid marking is 7. -/
theorem largest_valid_marking :
  (∃ (m : Marking 7), validMarking 7 m) ∧
  (∀ n > 7, ¬∃ (m : Marking n), validMarking n m) := by
  sorry

#check largest_valid_marking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_marking_l363_36391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l363_36373

theorem find_d : ∃ d : ℝ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 21 * n - 108 = 0) ∧ 
  (let f := d - ⌊d⌋; 5 * f^2 - 8 * f + 1 = 0) ∧
  (d = 4.8 - 0.2 * Real.sqrt 11 ∨ d = -8.2 - 0.2 * Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_d_l363_36373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_6_5_l363_36388

-- Define the vectors and point
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (1, 3)
def c : ℝ × ℝ := (2, 1)

-- Define the function to calculate the area of the triangle
noncomputable def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem triangle_area_is_6_5 :
  triangle_area c (c.1 + a.1, c.2 + a.2) (c.1 + b.1, c.2 + b.2) = 6.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_6_5_l363_36388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_sum_of_squares_l363_36386

theorem largest_divisor_of_sum_of_squares (a b c d e f : ℤ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = f^2) : 
  (∃ (k : ℕ), k = 24 ∧ (k : ℤ) ∣ (a * b * c * d * e * f) ∧ 
    ∀ (m : ℕ), m > k → ¬(∀ (x y z w v u : ℤ), 
      x^2 + y^2 + z^2 + w^2 + v^2 = u^2 → (m : ℤ) ∣ (x * y * z * w * v * u))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_sum_of_squares_l363_36386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_at_quarter_radius_l363_36301

open Real

/-- Given a sphere of radius r and an inscribed cone with an equilateral base,
    the function F(x) represents the difference in areas of sections made by a plane
    parallel to the base of the cone at distance x from the center of the sphere. -/
noncomputable def F (r x : ℝ) : ℝ := (2 * r^2 + 2 * r * x - 4 * x^2) / 3

/-- Theorem: The distance that maximizes the area difference is r/4 -/
theorem max_area_difference_at_quarter_radius (r : ℝ) (hr : r > 0) :
  ∃ (x : ℝ), x = r / 4 ∧ ∀ y, F r y ≤ F r x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_at_quarter_radius_l363_36301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_P_l363_36352

def M : Set ℕ := {x : ℕ | (x - 1)^2 < 4}
def P : Set ℕ := {0, 1, 2, 3}

theorem intersection_M_P : M ∩ P = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_P_l363_36352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_parameter_bound_l363_36307

/-- An inverse proportion function with parameter a -/
noncomputable def inverse_proportion (a : ℝ) : ℝ → ℝ := fun x ↦ (2 * a - 3) / x

/-- The condition that each branch of the graph increases as y increases with x -/
def increasing_branches (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ y₁ y₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ y₁ = f x₁ ∧ y₂ = f x₂ →
    (y₂ > y₁ ↔ x₂ > x₁)

/-- Theorem stating that if the inverse proportion function has increasing branches,
    then the parameter a must be less than 3/2 -/
theorem inverse_proportion_parameter_bound (a : ℝ) :
  increasing_branches (inverse_proportion a) → a < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_parameter_bound_l363_36307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l363_36308

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    where a > 0, b > 0, and one asymptote is y = x, is √2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ → ℝ), (∀ t, x t ^ 2 / a ^ 2 - y t ^ 2 / b ^ 2 = 1) ∧
                     (∃ c : ℝ, ∀ t, y t = x t + c)) →
  Real.sqrt ((a ^ 2 + b ^ 2) / (a ^ 2)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l363_36308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_left_age_l363_36383

/-- The age of the student who left the class -/
noncomputable def student_age : ℝ := 15.3

/-- The number of students in the class initially -/
def initial_students : ℕ := 45

/-- The initial average age of the students -/
def initial_average : ℝ := 14

/-- The new average age after one student leaves and teacher's age is included -/
def new_average : ℝ := 14.66

/-- The age of the teacher -/
def teacher_age : ℝ := 45

theorem student_left_age :
  (initial_students * initial_average - student_age + teacher_age) / initial_students = new_average ∧
  abs (student_age - 15.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_left_age_l363_36383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sams_gift_calculation_l363_36331

/-- Joan's initial seashell count -/
def initial_count : ℕ := 70

/-- Joan's final seashell count -/
def final_count : ℕ := 97

/-- The number of seashells Sam gave Joan -/
def sams_gift : ℕ := final_count - initial_count

/-- Theorem stating that Sam's gift is the difference between final and initial counts -/
theorem sams_gift_calculation : sams_gift = 27 := by
  rfl  -- reflexivity proves this automatically

#eval sams_gift  -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sams_gift_calculation_l363_36331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_five_2018_power_l363_36343

-- Define S as a function that calculates the sum of digits of a natural number
def S (n : ℕ) : ℕ := sorry

-- Define the property that S(n) ≡ n (mod 9) for all natural numbers n
axiom S_mod_9 (n : ℕ) : S n % 9 = n % 9

-- Define the function composition of S with itself k times
def S_iter : ℕ → (ℕ → ℕ)
  | 0 => id
  | k + 1 => S ∘ (S_iter k)

-- State the theorem
theorem S_five_2018_power : S_iter 5 (2018^(2018^2018)) = 7 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_five_2018_power_l363_36343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_difference_l363_36381

-- Define the system of equations
def satisfies_system (x y z : ℝ) : Prop :=
  (Real.sqrt 3 * Real.sin x = Real.tan y) ∧
  (2 * Real.sin y = 1 / Real.tan z) ∧
  (Real.sin z = 2 * Real.tan x)

-- Theorem statement
theorem min_cos_difference (x y z : ℝ) 
  (h : satisfies_system x y z) : 
  ∀ x' y' z' : ℝ, satisfies_system x' y' z' → 
  Real.cos x - Real.cos z ≥ -7 * Real.sqrt 2 / 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_difference_l363_36381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maximization_and_increase_l363_36390

/-- Represents the revenue multiplier as a function of price increase percentage -/
noncomputable def revenue_multiplier (x : ℝ) : ℝ := (1 + x / 100) * (1 - x / 100)

theorem revenue_maximization_and_increase {p n : ℝ} (hp : p > 0) (hn : n > 0) :
  /- (1) Revenue is maximized when x = 5 -/
  (∀ x : ℝ, 0 < x ∧ x ≤ 10 → revenue_multiplier x ≤ revenue_multiplier 5) ∧
  /- (2) Revenue increases when 0 < x < 5 -/
  (∀ x : ℝ, 0 < x ∧ x < 5 → revenue_multiplier x > 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maximization_and_increase_l363_36390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_three_digit_probability_l363_36345

def S : Finset ℕ := {1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Bool := n % 2 = 1

def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def valid_selection (a b c : ℕ) : Bool :=
  a ∈ S && b ∈ S && c ∈ S && a ≠ b && b ≠ c && a ≠ c

theorem odd_three_digit_probability :
  (Finset.filter (λ (abc : ℕ × ℕ × ℕ) => 
    valid_selection abc.1 abc.2.1 abc.2.2 &&
    is_odd (three_digit_number abc.1 abc.2.1 abc.2.2)) 
    (Finset.product S (Finset.product S S))).card /
  (Finset.filter (λ (abc : ℕ × ℕ × ℕ) => 
    valid_selection abc.1 abc.2.1 abc.2.2) 
    (Finset.product S (Finset.product S S))).card = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_three_digit_probability_l363_36345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_signed_quantity_interpretation_l363_36382

/-- Represents a signed quantity in kilograms -/
structure SignedQuantity where
  value : ℤ
  unit : String

/-- Interprets a SignedQuantity as an operation -/
def interpret (sq : SignedQuantity) : String :=
  if sq.value > 0 then
    s!"adding {sq.value.natAbs} {sq.unit}"
  else if sq.value < 0 then
    s!"subtracting {sq.value.natAbs} {sq.unit}"
  else
    s!"no change in {sq.unit}"

/-- The main theorem stating the interpretation of signed quantities -/
theorem signed_quantity_interpretation
  (pos_example : SignedQuantity)
  (neg_example : SignedQuantity)
  (h_pos : pos_example.value > 0)
  (h_neg : neg_example.value < 0)
  (h_unit : pos_example.unit = neg_example.unit)
  (h_pos_interp : interpret pos_example = s!"adding {pos_example.value.natAbs} {pos_example.unit}") :
  interpret neg_example = s!"subtracting {neg_example.value.natAbs} {neg_example.unit}" := by
  sorry

#eval interpret { value := 30, unit := "kg" }
#eval interpret { value := -70, unit := "kg" }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_signed_quantity_interpretation_l363_36382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_ratio_l363_36359

def total_amount : ℚ := 124600
def remaining_grandchildren : ℕ := 10
def amount_per_grandchild : ℚ := 6230

theorem shelby_ratio : 
  (total_amount - remaining_grandchildren * amount_per_grandchild) / total_amount = 1 / 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_ratio_l363_36359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l363_36327

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + 2)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, f x + f (a - 2*x) ≤ 1/2) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l363_36327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_six_over_pi_l363_36336

-- Define the radius of the circle
def radius : ℝ := 3

-- Define the number of semicircular arcs
def num_arcs : ℕ := 6

-- Define the area of the original circle
noncomputable def circle_area : ℝ := Real.pi * radius^2

-- Define the length of the rectangular figure
noncomputable def rect_length : ℝ := (num_arcs / 2 : ℝ) * radius

-- Define the width of the rectangular figure
def rect_width : ℝ := 2 * radius

-- Define the area of the rectangular figure
noncomputable def rect_area : ℝ := rect_length * rect_width

-- Theorem statement
theorem area_ratio_is_six_over_pi :
  rect_area / circle_area = 6 / Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_six_over_pi_l363_36336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y3_in_2x_minus_y_power_5_l363_36399

theorem coefficient_x2y3_in_2x_minus_y_power_5 :
  let n : ℕ := 5
  let term := fun (k : ℕ) => Int.ofNat (Nat.choose n k * (2^(n-k))) * ((-1 : ℤ)^k)
  term 3 = -40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y3_in_2x_minus_y_power_5_l363_36399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_b_wins_value_l363_36360

/-- The probability of Player B winning the championship in a best-of-five game system -/
def prob_b_wins (prob_a : ℚ) (prob_b : ℚ) : ℚ :=
  (Nat.choose 5 3) * prob_b^3 * prob_a^2 +
  (Nat.choose 5 4) * prob_b^4 * prob_a^1 +
  prob_b^5

/-- Theorem stating the probability of Player B winning the championship -/
theorem prob_b_wins_value :
  prob_b_wins (2/3) (1/3) = 17/81 := by
  sorry

#eval prob_b_wins (2/3) (1/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_b_wins_value_l363_36360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_log_l363_36358

theorem sum_reciprocals_equals_log (a b c : ℝ) 
  (h1 : (3 : ℝ)^a = 6) (h2 : (4 : ℝ)^b = 6) (h3 : (5 : ℝ)^c = 6) : 
  1/a + 1/b + 1/c = Real.log 60 / Real.log 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_log_l363_36358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l363_36323

open Real

-- Define the line l
def line_l (θ₀ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (ρ : ℝ), p.1 = ρ * cos θ₀ ∧ p.2 = ρ * sin θ₀}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - Real.sqrt 3)^2 = 1}

-- Theorem statement
theorem intersection_distance_range (θ₀ : ℝ) 
  (h1 : π/6 < θ₀) (h2 : θ₀ < π/3) :
  ∃ (M N : ℝ × ℝ), M ∈ line_l θ₀ ∧ M ∈ circle_C ∧
                   N ∈ line_l θ₀ ∧ N ∈ circle_C ∧
                   2 * Real.sqrt 3 < Real.sqrt (M.1^2 + M.2^2) + Real.sqrt (N.1^2 + N.2^2) ∧
                   Real.sqrt (M.1^2 + M.2^2) + Real.sqrt (N.1^2 + N.2^2) < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l363_36323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l363_36337

/-- Predicate indicating that a triangle with sides a, b, c is tangent to a sphere of radius r -/
def IsTangent (r a b c : ℝ) : Prop := sorry

/-- Calculates the distance from the center of a sphere to the plane of a triangle tangent to it -/
def DistanceToPlane (r a b c : ℝ) : ℝ := sorry

/-- Given a sphere and a triangle tangent to it, calculates the distance from the sphere's center to the triangle's plane -/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_sphere : r = 8) 
  (h_triangle : a = 13 ∧ b = 14 ∧ c = 15) (h_tangent : IsTangent r a b c) : 
  DistanceToPlane r a b c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triangle_distance_l363_36337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isha_pencil_theorem_l363_36379

/-- Represents a pencil with its initial and sharpened lengths -/
structure Pencil where
  initial_length : ℚ
  sharpened_length : ℚ

/-- Calculates the length sharpened off and the length of each half after breaking -/
def pencil_calculations (p : Pencil) : ℚ × ℚ :=
  let sharpened_off := p.initial_length - p.sharpened_length
  let half_length := p.sharpened_length / 2
  (sharpened_off, half_length)

/-- Theorem stating the correct calculations for Isha's pencil -/
theorem isha_pencil_theorem (p : Pencil) 
  (h1 : p.initial_length = 31)
  (h2 : p.sharpened_length = 14) :
  pencil_calculations p = (17, 7) := by
  sorry

#eval pencil_calculations ⟨31, 14⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isha_pencil_theorem_l363_36379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_is_five_l363_36338

def valid_list : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

def is_valid_pair (x y : Int) : Bool :=
  x ∈ valid_list && y > x && x + y = 3

def count_valid_pairs : Nat :=
  (valid_list.filter (λ x => valid_list.any (λ y => is_valid_pair x y))).length

theorem count_valid_pairs_is_five :
  count_valid_pairs = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_is_five_l363_36338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l363_36311

noncomputable def rotation_matrix (angle : Real) : Matrix (Fin 2) (Fin 2) Real :=
  ![![Real.cos angle, -Real.sin angle],
    ![Real.sin angle,  Real.cos angle]]

theorem smallest_rotation_power (n : Nat) : 
  n > 0 ∧ 
  rotation_matrix (135 * Real.pi / 180) ^ n = 1 ∧ 
  (∀ m : Nat, m > 0 ∧ m < n → rotation_matrix (135 * Real.pi / 180) ^ m ≠ 1) → 
  n = 8 := by
  sorry

#check smallest_rotation_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l363_36311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_upper_bound_l363_36354

/-- Rounds a natural number to the nearest power of 10 -/
def round_to_nearest_power_of_10 (k : ℕ) (power : ℕ) : ℕ :=
  sorry

/-- Performs n-1 rounds of rounding on a natural number k with n digits -/
def iterative_rounding (k : ℕ) (n : ℕ) : ℕ :=
  sorry

/-- Calculates the number of digits in a natural number -/
def num_digits (k : ℕ) : ℕ :=
  (Nat.digits 10 k).length

theorem rounding_upper_bound {k : ℕ} (h1 : 1 ≤ k) (h2 : k ≤ 199) :
  (iterative_rounding k (num_digits k)) < (18 * k) / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_upper_bound_l363_36354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_pigment_in_brown_paint_l363_36368

/-- Represents the composition of a paint -/
structure PaintComposition where
  blue : Float
  red : Float
  yellow : Float
  white : Float

/-- Represents the weight of each paint in the mixture -/
structure MixtureWeights where
  green : Float
  darkBlue : Float
  orange : Float

/-- Calculates the amount of red pigment in the brown paint -/
def redPigmentInBrownPaint (
  darkBlue : PaintComposition)
  (green : PaintComposition)
  (orange : PaintComposition)
  (brown : PaintComposition)
  (weights : MixtureWeights)
  (totalWeight : Float) : Float :=
  (darkBlue.red * weights.darkBlue + green.red * weights.green + orange.red * weights.orange) / totalWeight * totalWeight

/-- Theorem stating that the amount of red pigment in the brown paint is 10.8 grams -/
theorem red_pigment_in_brown_paint :
  let darkBlue := PaintComposition.mk 0.4 0.6 0 0
  let green := PaintComposition.mk 0.4 0 0.5 0.1
  let orange := PaintComposition.mk 0 0.5 0.4 0.1
  let brown := PaintComposition.mk 0.35 0.25 0.3 0.1
  let weights := MixtureWeights.mk 4 8 12
  let totalWeight := 24
  redPigmentInBrownPaint darkBlue green orange brown weights totalWeight = 10.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_pigment_in_brown_paint_l363_36368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_l363_36347

theorem angle_sum_is_pi (α β γ : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_cos_sum : Real.cos α + Real.cos β + Real.cos γ = 1 + 4 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2)) :
  α + β + γ = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_l363_36347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_difference_in_equation_l363_36384

theorem sign_difference_in_equation : ∃ (p m : ℤ), 
  (∃ (a b c : Bool), 
    (if a then 123 + 45 else 123 - 45) +
    (if b then (if a then 67 else -67) else (if a then -67 else 67)) +
    (if c then (if a = b then 89 else -89) else (if a = b then -89 else 89)) = 100 ∧
    p = (a.toNat + b.toNat + c.toNat) ∧
    m = (3 - (a.toNat + b.toNat + c.toNat))) ∧
  p - m = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_difference_in_equation_l363_36384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_l363_36344

theorem product_of_numbers (x y : ℝ) : 
  (x - y : ℝ) / (x + y : ℝ) = 1 / 8 →
  (x - y : ℝ) / (x * y : ℝ) = 1 / 40 →
  x - y = 2 → 
  x * y = 63 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_l363_36344
