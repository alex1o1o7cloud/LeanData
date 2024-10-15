import Mathlib

namespace NUMINAMATH_CALUDE_f_one_equals_four_l3115_311588

/-- The function f(x) = x^2 + ax - 3a - 9 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 3*a - 9

/-- The theorem stating that f(1) = 4 given the conditions -/
theorem f_one_equals_four (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : f a 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_one_equals_four_l3115_311588


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l3115_311514

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the midpoint octagon is 1/4 of the original octagon's area -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l3115_311514


namespace NUMINAMATH_CALUDE_no_factors_l3115_311515

def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 16

def factor1 (x : ℝ) : ℝ := x^2 - 4
def factor2 (x : ℝ) : ℝ := x + 2
def factor3 (x : ℝ) : ℝ := x^2 + 4*x + 4
def factor4 (x : ℝ) : ℝ := x^2 + 1

theorem no_factors :
  (∃ (x : ℝ), p x ≠ 0 ∧ factor1 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor2 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor3 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor4 x = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l3115_311515


namespace NUMINAMATH_CALUDE_worker_count_l3115_311511

theorem worker_count (total : ℕ) (extra_total : ℕ) (extra_contribution : ℕ) :
  total = 300000 →
  extra_total = 375000 →
  extra_contribution = 50 →
  ∃ n : ℕ, 
    n * (total / n) = total ∧
    n * (total / n + extra_contribution) = extra_total ∧
    n = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_worker_count_l3115_311511


namespace NUMINAMATH_CALUDE_total_song_requests_l3115_311581

/-- Represents the total number of song requests --/
def T : ℕ := 30

/-- Theorem stating that the total number of song requests is 30 --/
theorem total_song_requests :
  T = 30 ∧
  T = (1/2 : ℚ) * T + (1/6 : ℚ) * T + 5 + 2 + 1 + 2 :=
by sorry

end NUMINAMATH_CALUDE_total_song_requests_l3115_311581


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l3115_311563

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 80 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 65 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l3115_311563


namespace NUMINAMATH_CALUDE_ascending_order_l3115_311504

theorem ascending_order (a b c : ℝ) (ha : a = 60.7) (hb : b = 0.76) (hc : c = Real.log 0.76) :
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ascending_order_l3115_311504


namespace NUMINAMATH_CALUDE_area_of_triangle_RZX_l3115_311538

-- Define the square WXYZ
def Square (W X Y Z : ℝ × ℝ) : Prop :=
  -- Add conditions for a square here
  sorry

-- Define the area of a shape
def Area (shape : Set (ℝ × ℝ)) : ℝ :=
  sorry

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) (ratio : ℝ) : Prop :=
  -- P is on AB with AP:PB = ratio:(1-ratio)
  sorry

-- Define the midpoint of a line segment
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  -- M is the midpoint of AB
  sorry

theorem area_of_triangle_RZX 
  (W X Y Z : ℝ × ℝ)
  (P Q R : ℝ × ℝ)
  (h_square : Square W X Y Z)
  (h_area_WXYZ : Area {W, X, Y, Z} = 144)
  (h_P_on_YZ : PointOnSegment P Y Z (1/3))
  (h_Q_mid_WP : Midpoint Q W P)
  (h_R_mid_XP : Midpoint R X P)
  (h_area_YPRQ : Area {Y, P, R, Q} = 30)
  : Area {R, Z, X} = 24 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_RZX_l3115_311538


namespace NUMINAMATH_CALUDE_rotten_eggs_count_l3115_311503

theorem rotten_eggs_count (total : ℕ) (prob : ℚ) (h_total : total = 36) (h_prob : prob = 47619047619047615 / 10000000000000000) :
  ∃ (rotten : ℕ), rotten = 3 ∧
    (rotten : ℚ) / total * ((rotten : ℚ) - 1) / (total - 1) = prob :=
by sorry

end NUMINAMATH_CALUDE_rotten_eggs_count_l3115_311503


namespace NUMINAMATH_CALUDE_rabbit_speed_theorem_l3115_311531

/-- Given a rabbit's speed, double it, add 4, and double again -/
def rabbit_speed_operation (speed : ℕ) : ℕ :=
  ((speed * 2) + 4) * 2

/-- Theorem stating that the rabbit speed operation on 45 results in 188 -/
theorem rabbit_speed_theorem : rabbit_speed_operation 45 = 188 := by
  sorry

#eval rabbit_speed_operation 45  -- This will evaluate to 188

end NUMINAMATH_CALUDE_rabbit_speed_theorem_l3115_311531


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3115_311567

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3115_311567


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l3115_311575

theorem set_inclusion_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {0, 1} → 
  B = {-1, 0, a+3} → 
  A ⊆ B → 
  a = -2 := by sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l3115_311575


namespace NUMINAMATH_CALUDE_oranges_harvested_proof_l3115_311565

/-- The number of oranges harvested per day that are not discarded -/
def oranges_kept (sacks_harvested : ℕ) (sacks_discarded : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  (sacks_harvested - sacks_discarded) * oranges_per_sack

/-- Proof that the number of oranges harvested per day that are not discarded is 600 -/
theorem oranges_harvested_proof :
  oranges_kept 76 64 50 = 600 := by
  sorry

end NUMINAMATH_CALUDE_oranges_harvested_proof_l3115_311565


namespace NUMINAMATH_CALUDE_compare_negative_decimals_l3115_311524

theorem compare_negative_decimals : -3.3 < -3.14 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_decimals_l3115_311524


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3115_311562

theorem unique_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 3178 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3115_311562


namespace NUMINAMATH_CALUDE_unit_vectors_sum_squares_lower_bound_l3115_311551

theorem unit_vectors_sum_squares_lower_bound 
  (p q r : EuclideanSpace ℝ (Fin 3)) 
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hr : ‖r‖ = 1) : 
  ‖p + q‖^2 + ‖p + r‖^2 + ‖q + r‖^2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_sum_squares_lower_bound_l3115_311551


namespace NUMINAMATH_CALUDE_racecourse_length_l3115_311517

/-- The length of a racecourse where two runners A and B finish simultaneously,
    given that A runs twice as fast as B and B starts 42 meters ahead. -/
theorem racecourse_length : ℝ := by
  /- Let v be B's speed -/
  let v : ℝ := 1
  /- A's speed is twice B's speed -/
  let speed_A : ℝ := 2 * v
  /- B starts 42 meters ahead -/
  let head_start : ℝ := 42
  /- d is the length of the racecourse -/
  let d : ℝ := 84
  /- Time for A to finish the race -/
  let time_A : ℝ := d / speed_A
  /- Time for B to finish the race -/
  let time_B : ℝ := (d - head_start) / v
  /- A and B reach the finish line simultaneously -/
  have h : time_A = time_B := by sorry
  /- The racecourse length is 84 meters -/
  exact d

end NUMINAMATH_CALUDE_racecourse_length_l3115_311517


namespace NUMINAMATH_CALUDE_root_product_sum_l3115_311584

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (∀ x, Real.sqrt 2020 * x^3 - 4040 * x^2 + 4 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l3115_311584


namespace NUMINAMATH_CALUDE_total_pennies_thrown_l3115_311582

/-- The number of pennies thrown by each person -/
structure PennyThrowers where
  rachelle : ℕ
  gretchen : ℕ
  rocky : ℕ
  max : ℕ
  taylor : ℕ

/-- The conditions of the penny-throwing problem -/
def penny_throwing_conditions (pt : PennyThrowers) : Prop :=
  pt.rachelle = 720 ∧
  pt.gretchen = pt.rachelle / 2 ∧
  pt.rocky = pt.gretchen / 3 ∧
  pt.max = pt.rocky * 4 ∧
  pt.taylor = pt.max / 5

/-- The theorem stating that the total number of pennies thrown is 1776 -/
theorem total_pennies_thrown (pt : PennyThrowers) 
  (h : penny_throwing_conditions pt) : 
  pt.rachelle + pt.gretchen + pt.rocky + pt.max + pt.taylor = 1776 := by
  sorry


end NUMINAMATH_CALUDE_total_pennies_thrown_l3115_311582


namespace NUMINAMATH_CALUDE_trig_identity_l3115_311549

theorem trig_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3115_311549


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3115_311553

/-- Given a rectangle with perimeter 40, its maximum area is 100 -/
theorem rectangle_max_area :
  ∀ w l : ℝ,
  w > 0 → l > 0 →
  2 * (w + l) = 40 →
  ∀ w' l' : ℝ,
  w' > 0 → l' > 0 →
  2 * (w' + l') = 40 →
  w * l ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3115_311553


namespace NUMINAMATH_CALUDE_max_chain_length_is_optimal_l3115_311520

/-- Represents a triangular grid formed by dividing an equilateral triangle --/
structure TriangularGrid where
  n : ℕ
  total_triangles : ℕ := n^2

/-- Represents a chain of triangles in the grid --/
structure TriangleChain (grid : TriangularGrid) where
  length : ℕ
  is_valid : length ≤ grid.total_triangles

/-- The maximum length of a valid triangle chain in a given grid --/
def max_chain_length (grid : TriangularGrid) : ℕ :=
  grid.n^2 - grid.n + 1

/-- Theorem stating that the maximum chain length is n^2 - n + 1 --/
theorem max_chain_length_is_optimal (grid : TriangularGrid) :
  ∀ (chain : TriangleChain grid), chain.length ≤ max_chain_length grid :=
by sorry

end NUMINAMATH_CALUDE_max_chain_length_is_optimal_l3115_311520


namespace NUMINAMATH_CALUDE_min_value_theorem_l3115_311573

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  ∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3115_311573


namespace NUMINAMATH_CALUDE_probability_at_least_one_male_l3115_311533

theorem probability_at_least_one_male (male_count female_count : ℕ) 
  (h1 : male_count = 3) (h2 : female_count = 2) : 
  1 - (Nat.choose female_count 2 : ℚ) / (Nat.choose (male_count + female_count) 2 : ℚ) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_male_l3115_311533


namespace NUMINAMATH_CALUDE_bird_reserve_theorem_l3115_311576

/-- Represents the composition of birds in the Goshawk-Eurasian Nature Reserve -/
structure BirdReserve where
  total : ℝ
  hawks : ℝ
  paddyfield_warblers : ℝ
  kingfishers : ℝ

/-- The conditions of the bird reserve -/
def reserve_conditions (b : BirdReserve) : Prop :=
  b.hawks = 0.3 * b.total ∧
  b.paddyfield_warblers = 0.4 * (b.total - b.hawks) ∧
  b.kingfishers = 0.25 * b.paddyfield_warblers

/-- The theorem to be proved -/
theorem bird_reserve_theorem (b : BirdReserve) 
  (h : reserve_conditions b) : 
  (b.total - b.hawks - b.paddyfield_warblers - b.kingfishers) / b.total = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_bird_reserve_theorem_l3115_311576


namespace NUMINAMATH_CALUDE_min_value_theorem_l3115_311586

theorem min_value_theorem (a : ℝ) (h : a > 3) :
  a + 4 / (a - 3) ≥ 7 ∧ (a + 4 / (a - 3) = 7 ↔ a = 5) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3115_311586


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3115_311554

theorem complex_fraction_sum : 
  (Complex.I + 1)^2 / (Complex.I * 2 + 1) + (1 - Complex.I)^2 / (2 - Complex.I) = (6 - Complex.I * 2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3115_311554


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3115_311522

theorem imaginary_part_of_complex_number : 
  Complex.im (1 - Complex.I * Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3115_311522


namespace NUMINAMATH_CALUDE_world_not_ending_l3115_311594

theorem world_not_ending (n : ℕ) : ¬(∃ k : ℕ, (1 + n) = 11 * k ∧ (3 + 7 * n) = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_world_not_ending_l3115_311594


namespace NUMINAMATH_CALUDE_M_intersect_N_is_empty_l3115_311578

-- Define set M
def M : Set ℝ := {y | ∃ x, y = x + 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ (N.image Prod.snd) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_is_empty_l3115_311578


namespace NUMINAMATH_CALUDE_f_properties_l3115_311537

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a < 0 → ∀ x : ℝ, 0 < x → f a x > 0) ∧
  (0 < a → ∀ x : ℝ, 0 < x → (f a x > 0 ↔ x < 2*a)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3115_311537


namespace NUMINAMATH_CALUDE_first_hour_speed_l3115_311574

theorem first_hour_speed 
  (total_time : ℝ) 
  (average_speed : ℝ) 
  (second_part_time : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : total_time = 4) 
  (h2 : average_speed = 55) 
  (h3 : second_part_time = 3) 
  (h4 : second_part_speed = 60) : 
  ∃ (first_hour_speed : ℝ), first_hour_speed = 40 ∧ 
    average_speed * total_time = first_hour_speed * (total_time - second_part_time) + 
      second_part_speed * second_part_time :=
by sorry

end NUMINAMATH_CALUDE_first_hour_speed_l3115_311574


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3115_311556

theorem binomial_coefficient_two (n : ℕ) (h : n > 1) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3115_311556


namespace NUMINAMATH_CALUDE_total_sticks_is_326_l3115_311540

/-- The number of sticks needed for four rafts given specific conditions -/
def total_sticks : ℕ :=
  let simon := 45
  let gerry := (3 * simon) / 5
  let micky := simon + gerry + 15
  let darryl := 2 * micky - 7
  simon + gerry + micky + darryl

/-- Theorem stating that the total number of sticks needed is 326 -/
theorem total_sticks_is_326 : total_sticks = 326 := by
  sorry

end NUMINAMATH_CALUDE_total_sticks_is_326_l3115_311540


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3115_311543

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounces : ℕ) : ℝ :=
  3 * initialHeight - 2^(2 - bounces) * initialHeight

/-- Theorem: A ball dropped from 128 meters, bouncing to half its previous height each time,
    travels 383 meters after 9 bounces -/
theorem ball_bounce_distance :
  totalDistance 128 9 = 383 := by
  sorry

#eval totalDistance 128 9

end NUMINAMATH_CALUDE_ball_bounce_distance_l3115_311543


namespace NUMINAMATH_CALUDE_characterize_function_l3115_311559

open Set Function Real

-- Define the interval (1,∞)
def OpenOneInfty : Set ℝ := {x : ℝ | x > 1}

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ OpenOneInfty → y ∈ OpenOneInfty →
    (x^2 ≤ y ∧ y ≤ x^3) → ((f x)^2 ≤ f y ∧ f y ≤ (f x)^3)

-- The main theorem
theorem characterize_function :
  ∀ f : ℝ → ℝ, (∀ x, x ∈ OpenOneInfty → f x ∈ OpenOneInfty) →
    SatisfiesProperty f →
    ∃ k : ℝ, k > 0 ∧ ∀ x ∈ OpenOneInfty, f x = exp (k * log x) := by
  sorry

end NUMINAMATH_CALUDE_characterize_function_l3115_311559


namespace NUMINAMATH_CALUDE_min_m_plus_n_l3115_311552

/-- The set T of real numbers satisfying the given condition -/
def T : Set ℝ := Set.Iic 1

/-- The theorem stating the minimum value of m + n -/
theorem min_m_plus_n (m n : ℝ) (h_m : m > 1) (h_n : n > 1)
  (h_exists : ∃ x₀ : ℝ, ∀ x t : ℝ, t ∈ T → |x - 1| - |x - 2| ≥ t)
  (h_log : ∀ t ∈ T, Real.log m / Real.log 3 * Real.log n / Real.log 3 ≥ t) :
  m + n ≥ 6 ∧ ∃ m₀ n₀ : ℝ, m₀ > 1 ∧ n₀ > 1 ∧ m₀ + n₀ = 6 ∧
    (∀ t ∈ T, Real.log m₀ / Real.log 3 * Real.log n₀ / Real.log 3 ≥ t) :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l3115_311552


namespace NUMINAMATH_CALUDE_potato_bundle_price_l3115_311585

/-- Calculates the price of potato bundles given the harvest and sales information --/
theorem potato_bundle_price
  (potato_count : ℕ)
  (potato_bundle_size : ℕ)
  (carrot_count : ℕ)
  (carrot_bundle_size : ℕ)
  (carrot_bundle_price : ℚ)
  (total_revenue : ℚ)
  (h1 : potato_count = 250)
  (h2 : potato_bundle_size = 25)
  (h3 : carrot_count = 320)
  (h4 : carrot_bundle_size = 20)
  (h5 : carrot_bundle_price = 2)
  (h6 : total_revenue = 51) :
  (total_revenue - (carrot_count / carrot_bundle_size * carrot_bundle_price)) / (potato_count / potato_bundle_size) = 1.9 := by
sorry

end NUMINAMATH_CALUDE_potato_bundle_price_l3115_311585


namespace NUMINAMATH_CALUDE_jasons_treats_cost_l3115_311577

/-- Represents the quantity and price of a treat type -/
structure Treat where
  quantity : ℕ  -- quantity in dozens
  price : ℕ     -- price per dozen in dollars
  deriving Repr

/-- Calculates the total cost of treats -/
def totalCost (treats : List Treat) : ℕ :=
  treats.foldl (fun acc t => acc + t.quantity * t.price) 0

theorem jasons_treats_cost (cupcakes cookies brownies : Treat)
    (h1 : cupcakes = { quantity := 4, price := 10 })
    (h2 : cookies = { quantity := 3, price := 8 })
    (h3 : brownies = { quantity := 2, price := 12 }) :
    totalCost [cupcakes, cookies, brownies] = 88 := by
  sorry


end NUMINAMATH_CALUDE_jasons_treats_cost_l3115_311577


namespace NUMINAMATH_CALUDE_journey_average_speed_l3115_311525

/-- Proves that the average speed of a two-segment journey is 54.4 miles per hour -/
theorem journey_average_speed :
  let distance1 : ℝ := 200  -- miles
  let time1 : ℝ := 4.5      -- hours
  let distance2 : ℝ := 480  -- miles
  let time2 : ℝ := 8        -- hours
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 54.4      -- miles per hour
:= by sorry

end NUMINAMATH_CALUDE_journey_average_speed_l3115_311525


namespace NUMINAMATH_CALUDE_eighth_term_of_happy_sequence_l3115_311516

def happy_sequence (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / 2^n

theorem eighth_term_of_happy_sequence :
  happy_sequence 8 = 1/32 := by sorry

end NUMINAMATH_CALUDE_eighth_term_of_happy_sequence_l3115_311516


namespace NUMINAMATH_CALUDE_existence_of_m_l3115_311596

theorem existence_of_m (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, m * a > m * b :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l3115_311596


namespace NUMINAMATH_CALUDE_tan_one_iff_quarter_pi_plus_multiple_pi_l3115_311548

theorem tan_one_iff_quarter_pi_plus_multiple_pi (x : ℝ) : 
  Real.tan x = 1 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_one_iff_quarter_pi_plus_multiple_pi_l3115_311548


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_one_l3115_311541

theorem intersection_nonempty_implies_a_less_than_one (a : ℝ) : 
  let M := {x : ℝ | x ≤ 1}
  let P := {x : ℝ | x > a}
  (M ∩ P).Nonempty → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_less_than_one_l3115_311541


namespace NUMINAMATH_CALUDE_min_width_proof_l3115_311536

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 10

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 15

/-- The area of the rectangular region -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 200 → w ≥ min_width) ∧
  (area min_width ≥ 200) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l3115_311536


namespace NUMINAMATH_CALUDE_max_subsets_of_N_l3115_311528

/-- The set M -/
def M : Finset ℕ := {0, 2, 3, 7}

/-- The set N -/
def N : Finset ℕ := Finset.image (λ (p : ℕ × ℕ) => p.1 * p.2) (M.product M)

/-- Theorem: The maximum number of subsets of N is 128 -/
theorem max_subsets_of_N : Finset.card (Finset.powerset N) = 128 := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_of_N_l3115_311528


namespace NUMINAMATH_CALUDE_acme_soup_words_count_l3115_311500

/-- The number of possible words of length n formed from a set of k distinct letters,
    where each letter appears at least n times. -/
def word_count (n k : ℕ) : ℕ := k^n

/-- The specific case for 6-letter words formed from 6 distinct letters. -/
def acme_soup_words : ℕ := word_count 6 6

theorem acme_soup_words_count :
  acme_soup_words = 46656 := by
  sorry

end NUMINAMATH_CALUDE_acme_soup_words_count_l3115_311500


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_equals_negative_one_l3115_311523

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 1

-- State the theorem
theorem decreasing_quadratic_implies_a_equals_negative_one :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y < 2 → f a x > f a y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_equals_negative_one_l3115_311523


namespace NUMINAMATH_CALUDE_milk_water_ratio_change_l3115_311590

/-- Proves that adding 60 litres of water to a 60-litre mixture with initial milk to water ratio of 2:1 results in a new ratio of 1:2 -/
theorem milk_water_ratio_change (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) :
  initial_volume = 60 →
  initial_milk_ratio = 2 →
  initial_water_ratio = 1 →
  added_water = 60 →
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let new_water := initial_water + added_water
  let new_milk_ratio := initial_milk / new_water
  let new_water_ratio := new_water / new_water
  new_milk_ratio = 1 ∧ new_water_ratio = 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_change_l3115_311590


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_l3115_311580

def square_of_1222222221 : ℕ := 1493822537037038241

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_square :
  sum_of_digits square_of_1222222221 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_l3115_311580


namespace NUMINAMATH_CALUDE_sin_seventeen_pi_fourths_l3115_311544

theorem sin_seventeen_pi_fourths : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seventeen_pi_fourths_l3115_311544


namespace NUMINAMATH_CALUDE_least_product_for_divisibility_least_product_for_cross_divisibility_l3115_311513

theorem least_product_for_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a.val^a.val * b.val^b.val = 2000 * k) →
  (∀ c d : ℕ+, (∃ l : ℕ, c.val^c.val * d.val^d.val = 2000 * l) → c.val * d.val ≥ 10) ∧
  (∃ m n : ℕ+, m.val * n.val = 10 ∧ ∃ p : ℕ, m.val^m.val * n.val^n.val = 2000 * p) :=
sorry

theorem least_product_for_cross_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a.val^b.val * b.val^a.val = 2000 * k) →
  (∀ c d : ℕ+, (∃ l : ℕ, c.val^d.val * d.val^c.val = 2000 * l) → c.val * d.val ≥ 20) ∧
  (∃ m n : ℕ+, m.val * n.val = 20 ∧ ∃ p : ℕ, m.val^n.val * n.val^m.val = 2000 * p) :=
sorry

end NUMINAMATH_CALUDE_least_product_for_divisibility_least_product_for_cross_divisibility_l3115_311513


namespace NUMINAMATH_CALUDE_lcm_24_36_42_l3115_311518

theorem lcm_24_36_42 : Nat.lcm (Nat.lcm 24 36) 42 = 504 := by sorry

end NUMINAMATH_CALUDE_lcm_24_36_42_l3115_311518


namespace NUMINAMATH_CALUDE_find_5b_l3115_311570

theorem find_5b (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_find_5b_l3115_311570


namespace NUMINAMATH_CALUDE_proposition_truth_l3115_311547

theorem proposition_truth (p q : Prop) (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l3115_311547


namespace NUMINAMATH_CALUDE_parabola_m_minus_one_opens_downward_l3115_311592

/-- A parabola y = ax^2 opens downward if and only if a < 0 -/
axiom parabola_opens_downward (a : ℝ) : (∀ x y : ℝ, y = a * x^2) → (∀ x : ℝ, a * x^2 ≤ 0) ↔ a < 0

/-- For the parabola y = (m-1)x^2 to open downward, m must be less than 1 -/
theorem parabola_m_minus_one_opens_downward (m : ℝ) :
  (∀ x y : ℝ, y = (m - 1) * x^2) → (∀ x : ℝ, (m - 1) * x^2 ≤ 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_m_minus_one_opens_downward_l3115_311592


namespace NUMINAMATH_CALUDE_all_divisors_end_in_one_l3115_311527

theorem all_divisors_end_in_one (n : ℕ+) :
  ∀ d : ℕ, d > 0 → d ∣ ((10^(5^n.val) - 1) / 9) → d % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_all_divisors_end_in_one_l3115_311527


namespace NUMINAMATH_CALUDE_five_consecutive_not_square_l3115_311501

theorem five_consecutive_not_square (n : ℤ) : ¬∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_not_square_l3115_311501


namespace NUMINAMATH_CALUDE_triangle_count_l3115_311519

/-- A triangle with integral side lengths. -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if the given integers form a valid triangle. -/
def is_valid_triangle (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Check if the triangle has a perimeter of 9. -/
def has_perimeter_9 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 9

/-- Two triangles are considered different if they are not congruent. -/
def are_different (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

theorem triangle_count : 
  ∃ (t1 t2 : IntTriangle), 
    is_valid_triangle t1 ∧ 
    is_valid_triangle t2 ∧ 
    has_perimeter_9 t1 ∧ 
    has_perimeter_9 t2 ∧ 
    are_different t1 t2 ∧
    (∀ (t3 : IntTriangle), 
      is_valid_triangle t3 → 
      has_perimeter_9 t3 → 
      (t3 = t1 ∨ t3 = t2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l3115_311519


namespace NUMINAMATH_CALUDE_parabola_chord_theorem_l3115_311558

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 -/
def parabola (p : Point) : Prop := p.y = p.x^2

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Checks if a point divides a line segment in a given ratio -/
def divides_in_ratio (p1 p2 p : Point) (m n : ℝ) : Prop :=
  n * (p.x - p1.x) = m * (p2.x - p.x) ∧ n * (p.y - p1.y) = m * (p2.y - p.y)

theorem parabola_chord_theorem (A B C : Point) :
  parabola A ∧ parabola B ∧  -- A and B lie on the parabola
  C.x = 0 ∧ C.y = 15 ∧  -- C is on y-axis with y-coordinate 15
  collinear A B C ∧  -- A, B, and C are collinear
  divides_in_ratio A B C 5 3 →  -- C divides AB in ratio 5:3
  ((A.x = -5 ∧ B.x = 3) ∨ (A.x = 5 ∧ B.x = -3)) := by sorry

end NUMINAMATH_CALUDE_parabola_chord_theorem_l3115_311558


namespace NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3115_311542

def cost_problem (apple banana cantaloupe date : ℝ) : Prop :=
  apple + banana + cantaloupe + date = 40 ∧
  date = 3 * apple ∧
  banana = cantaloupe - 2

theorem banana_cantaloupe_cost 
  (apple banana cantaloupe date : ℝ)
  (h : cost_problem apple banana cantaloupe date) :
  banana + cantaloupe = 20 := by
sorry

end NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3115_311542


namespace NUMINAMATH_CALUDE_correct_statements_about_squares_l3115_311526

theorem correct_statements_about_squares :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧
  (∀ x : ℝ, x < -1 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_about_squares_l3115_311526


namespace NUMINAMATH_CALUDE_junior_score_l3115_311564

theorem junior_score (total : ℕ) (junior_score : ℝ) : 
  total > 0 →
  let junior_count : ℝ := 0.2 * total
  let senior_count : ℝ := 0.8 * total
  let overall_avg : ℝ := 86
  let senior_avg : ℝ := 85
  (junior_count * junior_score + senior_count * senior_avg) / total = overall_avg →
  junior_score = 90 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l3115_311564


namespace NUMINAMATH_CALUDE_complex_magnitude_l3115_311546

theorem complex_magnitude (z : ℂ) (h : z - 2 + Complex.I = 1) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3115_311546


namespace NUMINAMATH_CALUDE_pet_store_birds_l3115_311534

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 4

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 8

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 40 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l3115_311534


namespace NUMINAMATH_CALUDE_right_triangle_least_side_l3115_311591

theorem right_triangle_least_side (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → min a (min b c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_least_side_l3115_311591


namespace NUMINAMATH_CALUDE_harveys_steaks_l3115_311566

theorem harveys_steaks (initial_steaks : ℕ) 
  (h1 : initial_steaks - 17 = 12) 
  (h2 : 17 ≥ 4) : initial_steaks = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_harveys_steaks_l3115_311566


namespace NUMINAMATH_CALUDE_remaining_battery_life_is_eight_hours_l3115_311595

/-- Represents the battery life of a phone -/
structure PhoneBattery where
  inactiveLife : ℝ  -- Battery life when not in use (in hours)
  activeLife : ℝ    -- Battery life when used constantly (in hours)

/-- Calculates the remaining battery life -/
def remainingBatteryLife (battery : PhoneBattery) 
  (usedTime : ℝ)     -- Time the phone has been used (in hours)
  (totalTime : ℝ)    -- Total time since last charge (in hours)
  : ℝ :=
  sorry

/-- Theorem: Given the conditions, the remaining battery life is 8 hours -/
theorem remaining_battery_life_is_eight_hours 
  (battery : PhoneBattery)
  (h1 : battery.inactiveLife = 18)
  (h2 : battery.activeLife = 2)
  (h3 : remainingBatteryLife battery 0.5 6 = 8) :
  ∃ (t : ℝ), t = 8 ∧ remainingBatteryLife battery 0.5 6 = t := by
  sorry

end NUMINAMATH_CALUDE_remaining_battery_life_is_eight_hours_l3115_311595


namespace NUMINAMATH_CALUDE_john_supermarket_spending_l3115_311530

def supermarket_spending (total : ℚ) : Prop :=
  let fruits_veg := (1 : ℚ) / 5 * total
  let meat := (1 : ℚ) / 3 * total
  let bakery := (1 : ℚ) / 10 * total
  let dairy := (1 : ℚ) / 6 * total
  let candy_magazine := total - (fruits_veg + meat + bakery + dairy)
  let magazine := (15 : ℚ) / 4  -- $3.75 as a rational number
  candy_magazine = (29 : ℚ) / 2 ∧  -- $14.50 as a rational number
  candy_magazine - magazine = (43 : ℚ) / 4  -- $10.75 as a rational number

theorem john_supermarket_spending :
  ∃ (total : ℚ), supermarket_spending total ∧ total = (145 : ℚ) / 2 :=
sorry

end NUMINAMATH_CALUDE_john_supermarket_spending_l3115_311530


namespace NUMINAMATH_CALUDE_katyas_age_l3115_311529

def insert_zero (n : ℕ) : ℕ :=
  (n / 10) * 100 + (n % 10)

theorem katyas_age :
  ∃! n : ℕ, n ≥ 10 ∧ n < 100 ∧ 6 * n = insert_zero n ∧ n = 18 :=
by sorry

end NUMINAMATH_CALUDE_katyas_age_l3115_311529


namespace NUMINAMATH_CALUDE_sheep_to_horse_ratio_l3115_311539

theorem sheep_to_horse_ratio :
  let horse_food_per_day : ℕ := 230
  let total_horse_food : ℕ := 12880
  let num_sheep : ℕ := 16
  let num_horses : ℕ := total_horse_food / horse_food_per_day
  (num_sheep : ℚ) / (num_horses : ℚ) = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_sheep_to_horse_ratio_l3115_311539


namespace NUMINAMATH_CALUDE_store_rooms_problem_l3115_311545

/-- The number of rooms in Li Sangong's store -/
def num_rooms : ℕ := 8

/-- The total number of people visiting the store -/
def total_people : ℕ := 7 * num_rooms + 7

theorem store_rooms_problem :
  (total_people = 7 * num_rooms + 7) ∧
  (total_people = 9 * (num_rooms - 1)) ∧
  (num_rooms = 8) := by
  sorry

end NUMINAMATH_CALUDE_store_rooms_problem_l3115_311545


namespace NUMINAMATH_CALUDE_instantaneous_velocity_one_l3115_311507

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 * t^2 - 2

-- Define the instantaneous velocity (derivative of s with respect to t)
def v (t : ℝ) : ℝ := 6 * t

-- Theorem: The time at which the instantaneous velocity is 1 is 1/6
theorem instantaneous_velocity_one (t : ℝ) : v t = 1 ↔ t = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_one_l3115_311507


namespace NUMINAMATH_CALUDE_fred_car_washing_earnings_l3115_311532

/-- Fred's earnings from various activities -/
structure FredEarnings where
  total : ℕ
  newspaper : ℕ
  car_washing : ℕ

/-- Theorem stating that Fred's car washing earnings are 74 dollars -/
theorem fred_car_washing_earnings (e : FredEarnings) 
  (h1 : e.total = 90)
  (h2 : e.newspaper = 16)
  (h3 : e.total = e.newspaper + e.car_washing) :
  e.car_washing = 74 := by
  sorry

end NUMINAMATH_CALUDE_fred_car_washing_earnings_l3115_311532


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3115_311509

theorem not_sufficient_nor_necessary : ¬(∀ x : ℝ, (x - 2) * (x - 1) > 0 → (x - 2 > 0 ∨ x - 1 > 0)) ∧
                                       ¬(∀ x : ℝ, (x - 2 > 0 ∨ x - 1 > 0) → (x - 2) * (x - 1) > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3115_311509


namespace NUMINAMATH_CALUDE_amount_in_paise_l3115_311598

theorem amount_in_paise : 
  let a : ℝ := 190
  let percentage : ℝ := 0.5
  let amount_in_rupees : ℝ := percentage / 100 * a
  let paise_per_rupee : ℕ := 100
  ⌊amount_in_rupees * paise_per_rupee⌋ = 95 := by sorry

end NUMINAMATH_CALUDE_amount_in_paise_l3115_311598


namespace NUMINAMATH_CALUDE_train_length_l3115_311593

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 8 seconds,
    and the platform length is 1162.5 meters, prove that the length of the train is 300 meters. -/
theorem train_length (crossing_platform_time : ℝ) (crossing_pole_time : ℝ) (platform_length : ℝ)
  (h1 : crossing_platform_time = 39)
  (h2 : crossing_pole_time = 8)
  (h3 : platform_length = 1162.5) :
  ∃ (train_length : ℝ), train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3115_311593


namespace NUMINAMATH_CALUDE_number_difference_l3115_311505

theorem number_difference (a b : ℕ) : 
  a + b = 20000 → 7 * a = b → b - a = 15000 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3115_311505


namespace NUMINAMATH_CALUDE_milk_remaining_l3115_311579

theorem milk_remaining (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) : 
  initial_milk = 5 → given_milk = 18/7 → remaining_milk = initial_milk - given_milk → remaining_milk = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l3115_311579


namespace NUMINAMATH_CALUDE_product_sum_fractions_l3115_311521

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l3115_311521


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3115_311583

theorem polynomial_divisibility (m n : ℕ) :
  ∃ q : Polynomial ℤ, (X^2 + X + 1) * q = X^(3*m+2) + (-X^2 - 1)^(3*n+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3115_311583


namespace NUMINAMATH_CALUDE_smallest_in_S_l3115_311568

def S : Set ℝ := {1, -2, -1.7, 0, Real.pi}

theorem smallest_in_S : ∀ x ∈ S, -2 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_in_S_l3115_311568


namespace NUMINAMATH_CALUDE_roof_area_difference_l3115_311550

/-- Proves the difference in area between two rectangular roofs -/
theorem roof_area_difference (w : ℝ) (h1 : w > 0) (h2 : 4 * w * w = 784) : 
  5 * w * w - 4 * w * w = 196 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_difference_l3115_311550


namespace NUMINAMATH_CALUDE_base_sequences_count_l3115_311508

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of different base sequences that can be formed from one base A, two bases C, and three bases G --/
def base_sequences : ℕ :=
  choose 6 1 * choose 5 2 * choose 3 3

theorem base_sequences_count : base_sequences = 60 := by sorry

end NUMINAMATH_CALUDE_base_sequences_count_l3115_311508


namespace NUMINAMATH_CALUDE_number_of_book_pairs_l3115_311560

/-- Represents the number of books in each genre --/
structure BookCollection where
  mystery : Nat
  fantasy : Nat
  biography : Nat

/-- Represents the condition that "Mystery Masterpiece" must be included --/
def mustIncludeMysteryMasterpiece : Bool := true

/-- Calculates the number of possible book pairs --/
def calculatePossiblePairs (books : BookCollection) : Nat :=
  books.fantasy + books.biography

/-- Theorem stating the number of possible book pairs --/
theorem number_of_book_pairs :
  let books : BookCollection := ⟨4, 3, 3⟩
  calculatePossiblePairs books = 6 := by sorry

end NUMINAMATH_CALUDE_number_of_book_pairs_l3115_311560


namespace NUMINAMATH_CALUDE_children_retaking_test_l3115_311571

theorem children_retaking_test (total : Float) (passed : Float) 
  (h1 : total = 698.0) (h2 : passed = 105.0) : 
  total - passed = 593.0 := by
  sorry

end NUMINAMATH_CALUDE_children_retaking_test_l3115_311571


namespace NUMINAMATH_CALUDE_notebook_purchase_difference_l3115_311589

theorem notebook_purchase_difference (price : ℚ) (marie_count jake_count : ℕ) : 
  price > (1/4 : ℚ) →
  price * marie_count = (15/4 : ℚ) →
  price * jake_count = 5 →
  jake_count - marie_count = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_purchase_difference_l3115_311589


namespace NUMINAMATH_CALUDE_smallest_whole_multiple_651_l3115_311506

def is_whole_multiple (n m : ℕ) : Prop := n % m = 0

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def P (n : ℕ) : ℕ := max ((n / 100) * 10 + (n / 10) % 10) (max ((n / 100) * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

def Q (n : ℕ) : ℕ := min ((n / 100) * 10 + (n / 10) % 10) (min ((n / 100) * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

theorem smallest_whole_multiple_651 : 
  ∃ (A : ℕ), 
    100 ≤ A ∧ A < 1000 ∧ 
    is_whole_multiple A 12 ∧
    digit_sum A = 12 ∧ 
    (A % 10 < (A / 10) % 10) ∧ ((A / 10) % 10 < A / 100) ∧
    ((P A + Q A) % 2 = 0) ∧
    (∀ (B : ℕ), 
      100 ≤ B ∧ B < 1000 ∧ 
      is_whole_multiple B 12 ∧
      digit_sum B = 12 ∧ 
      (B % 10 < (B / 10) % 10) ∧ ((B / 10) % 10 < B / 100) ∧
      ((P B + Q B) % 2 = 0) →
      A ≤ B) ∧
    A = 651 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_multiple_651_l3115_311506


namespace NUMINAMATH_CALUDE_polynomial_factor_sum_l3115_311510

theorem polynomial_factor_sum (a b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^3 + a*x^2 + b*x + 8 = (x + 1) * (x + 2) * (x + c)) → 
  a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_sum_l3115_311510


namespace NUMINAMATH_CALUDE_math_class_size_l3115_311535

theorem math_class_size (total_students : ℕ) (both_subjects : ℕ) :
  total_students = 75 →
  both_subjects = 15 →
  ∃ (math_only physics_only : ℕ),
    total_students = math_only + physics_only + both_subjects ∧
    math_only + both_subjects = 2 * (physics_only + both_subjects) →
  math_only + both_subjects = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_class_size_l3115_311535


namespace NUMINAMATH_CALUDE_complex_simplification_l3115_311561

theorem complex_simplification :
  (5 - 3*Complex.I) + (-2 + 6*Complex.I) - (7 - 2*Complex.I) = -4 + 5*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3115_311561


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3115_311557

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 2) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3115_311557


namespace NUMINAMATH_CALUDE_unit_vector_of_AB_l3115_311502

/-- Given a plane vector AB = (-1, 2), prove that its unit vector is (-√5/5, 2√5/5) -/
theorem unit_vector_of_AB (AB : ℝ × ℝ) (h : AB = (-1, 2)) :
  let magnitude := Real.sqrt ((AB.1)^2 + (AB.2)^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (-Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_of_AB_l3115_311502


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l3115_311555

-- Define the square and triangle
def square_side_length : ℝ := 16
def triangle_leg_length : ℝ := 8

-- Define the theorem
theorem perimeter_of_modified_square :
  let square_perimeter := 4 * square_side_length
  let triangle_hypotenuse := Real.sqrt (2 * triangle_leg_length ^ 2)
  let new_figure_perimeter := square_perimeter - triangle_leg_length + triangle_hypotenuse
  new_figure_perimeter = 64 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_modified_square_l3115_311555


namespace NUMINAMATH_CALUDE_triangle_area_l3115_311587

/-- The area of a triangle with vertices (5, -2), (10, 5), and (5, 5) is 17.5 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (5, -2)
  let v2 : ℝ × ℝ := (10, 5)
  let v3 : ℝ × ℝ := (5, 5)
  let area := (1/2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))
  area = 17.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3115_311587


namespace NUMINAMATH_CALUDE_min_value_expression_l3115_311599

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 ∧
  ∀ (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0),
    x^2 + y^2 + z^2 + 1/x^2 + y/x + z/y ≥ min ∧
    ∃ (a' b' c' : ℝ) (ha' : a' ≠ 0) (hb' : b' ≠ 0) (hc' : c' ≠ 0),
      a'^2 + b'^2 + c'^2 + 1/a'^2 + b'/a' + c'/b' = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3115_311599


namespace NUMINAMATH_CALUDE_probability_two_girls_five_tickets_l3115_311512

/-- The probability of selecting 2 girls when drawing 5 tickets from a group of 25 students, of which 10 are girls, is 195/506. -/
theorem probability_two_girls_five_tickets (total_students : Nat) (girls : Nat) (tickets : Nat) :
  total_students = 25 →
  girls = 10 →
  tickets = 5 →
  (Nat.choose girls 2 * Nat.choose (total_students - girls) (tickets - 2)) / Nat.choose total_students tickets = 195 / 506 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_five_tickets_l3115_311512


namespace NUMINAMATH_CALUDE_exists_inner_sum_greater_than_outer_sum_l3115_311597

/-- Represents a triangular pyramid (tetrahedron) --/
structure TriangularPyramid where
  base_edge_length : ℝ
  lateral_edge_length : ℝ

/-- Calculates the sum of edge lengths of a triangular pyramid --/
def sum_of_edges (pyramid : TriangularPyramid) : ℝ :=
  3 * pyramid.base_edge_length + 3 * pyramid.lateral_edge_length

/-- Represents two triangular pyramids with a common base, where one is inside the other --/
structure NestedPyramids where
  outer : TriangularPyramid
  inner : TriangularPyramid
  inner_inside_outer : inner.base_edge_length = outer.base_edge_length
  inner_lateral_edge_shorter : inner.lateral_edge_length < outer.lateral_edge_length

/-- Theorem: There exist nested pyramids where the sum of edges of the inner pyramid
    is greater than the sum of edges of the outer pyramid --/
theorem exists_inner_sum_greater_than_outer_sum :
  ∃ (np : NestedPyramids), sum_of_edges np.inner > sum_of_edges np.outer := by
  sorry


end NUMINAMATH_CALUDE_exists_inner_sum_greater_than_outer_sum_l3115_311597


namespace NUMINAMATH_CALUDE_study_tour_students_l3115_311572

/-- Represents the number of students participating in the study tour. -/
def num_students : ℕ := 46

/-- Represents the number of dormitories. -/
def num_dormitories : ℕ := 6

theorem study_tour_students :
  (∃ (n : ℕ), n = num_dormitories ∧
    6 * n + 10 = num_students ∧
    8 * (n - 1) + 4 < num_students ∧
    num_students < 8 * (n - 1) + 8) :=
by sorry

end NUMINAMATH_CALUDE_study_tour_students_l3115_311572


namespace NUMINAMATH_CALUDE_town_population_l3115_311569

theorem town_population (total_population : ℕ) 
  (h1 : total_population < 6000)
  (h2 : ∃ (boys girls : ℕ), girls = (11 * boys) / 10 ∧ boys + girls = total_population * 10 / 21)
  (h3 : ∃ (women men : ℕ), men = (23 * women) / 20 ∧ women + men = total_population * 20 / 43)
  (h4 : ∃ (children adults : ℕ), children = (6 * adults) / 5 ∧ children + adults = total_population) :
  total_population = 3311 := by
sorry

end NUMINAMATH_CALUDE_town_population_l3115_311569
