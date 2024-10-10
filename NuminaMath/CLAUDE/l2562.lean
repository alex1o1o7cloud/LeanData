import Mathlib

namespace vector_parallel_implies_k_equals_one_l2562_256224

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 7)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

-- Theorem statement
theorem vector_parallel_implies_k_equals_one (k : ℝ) :
  parallel (a.1 + 2 * (c k).1, a.2 + 2 * (c k).2) b → k = 1 := by
  sorry

end vector_parallel_implies_k_equals_one_l2562_256224


namespace tiling_condition_l2562_256267

/-- Represents a tile type with its dimensions -/
inductive TileType
  | square : TileType  -- 2 × 2 tile
  | rectangle : TileType  -- 3 × 1 tile

/-- Calculates the area of a tile -/
def tileArea (t : TileType) : ℕ :=
  match t with
  | TileType.square => 4
  | TileType.rectangle => 3

/-- Represents a floor tiling with square and rectangle tiles -/
structure Tiling (n : ℕ) where
  numTiles : ℕ
  complete : n * n = numTiles * (tileArea TileType.square + tileArea TileType.rectangle)

/-- Theorem: A square floor of size n × n can be tiled with an equal number of 2 × 2 and 3 × 1 tiles
    if and only if n is divisible by 7 -/
theorem tiling_condition (n : ℕ) :
  (∃ t : Tiling n, True) ↔ ∃ k : ℕ, n = 7 * k :=
by sorry

end tiling_condition_l2562_256267


namespace salary_for_may_l2562_256231

/-- Proves that the salary for May is 3600, given the average salaries for two sets of four months and the salary for January. -/
theorem salary_for_may (jan feb mar apr may : ℝ) : 
  (jan + feb + mar + apr) / 4 = 8000 →
  (feb + mar + apr + may) / 4 = 8900 →
  jan = 2900 →
  may = 3600 := by
  sorry

end salary_for_may_l2562_256231


namespace gcd_35_and_number_between_65_and_75_l2562_256284

theorem gcd_35_and_number_between_65_and_75 :
  ∃! n : ℕ, 65 < n ∧ n < 75 ∧ Nat.gcd 35 n = 7 :=
by
  -- The proof goes here
  sorry

end gcd_35_and_number_between_65_and_75_l2562_256284


namespace mitchs_weekly_earnings_l2562_256205

/-- Mitch's weekly earnings calculation --/
theorem mitchs_weekly_earnings : 
  let weekday_hours : ℕ := 5
  let weekend_hours : ℕ := 3
  let weekday_rate : ℕ := 3
  let weekend_rate : ℕ := 2 * weekday_rate
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2

  weekdays * weekday_hours * weekday_rate + 
  weekend_days * weekend_hours * weekend_rate = 111 := by
  sorry

end mitchs_weekly_earnings_l2562_256205


namespace integral_reciprocal_x_from_one_over_e_to_e_l2562_256279

open Real MeasureTheory

theorem integral_reciprocal_x_from_one_over_e_to_e :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), (1 / x) = 2 := by
  sorry

end integral_reciprocal_x_from_one_over_e_to_e_l2562_256279


namespace negative_integer_solutions_l2562_256240

def satisfies_inequalities (x : ℤ) : Prop :=
  3 * x - 2 ≥ 2 * x - 5 ∧ x / 2 - (x - 2) / 3 < 1 / 2

theorem negative_integer_solutions :
  {x : ℤ | x < 0 ∧ satisfies_inequalities x} = {-3, -2} :=
by sorry

end negative_integer_solutions_l2562_256240


namespace minimum_guests_l2562_256241

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (min_guests : ℕ) :
  total_food = 406 →
  max_per_guest = 2.5 →
  min_guests = 163 →
  (↑min_guests : ℝ) * max_per_guest ≥ total_food ∧
  ∀ n : ℕ, (↑n : ℝ) * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end minimum_guests_l2562_256241


namespace area_of_region_is_10_625_l2562_256217

/-- The lower boundary function of the region -/
def lower_boundary (x : ℝ) : ℝ := |x - 4|

/-- The upper boundary function of the region -/
def upper_boundary (x : ℝ) : ℝ := 5 - |x - 2|

/-- The region in the xy-plane -/
def region : Set (ℝ × ℝ) :=
  {p | lower_boundary p.1 ≤ p.2 ∧ p.2 ≤ upper_boundary p.1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_region_is_10_625 : area_of_region = 10.625 := by sorry

end area_of_region_is_10_625_l2562_256217


namespace hundred_brick_tower_heights_l2562_256243

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : Nat
  width : Nat
  height : Nat

/-- Calculates the number of different tower heights achievable -/
def towerHeights (brickCount : Nat) (dimensions : BrickDimensions) : Nat :=
  sorry

/-- The main theorem stating the number of different tower heights -/
theorem hundred_brick_tower_heights :
  let brickDims : BrickDimensions := { length := 3, width := 11, height := 18 }
  towerHeights 100 brickDims = 1404 := by
  sorry

end hundred_brick_tower_heights_l2562_256243


namespace calculation_proof_l2562_256296

theorem calculation_proof : (π - 3.15) ^ 0 * (-1) ^ 2023 - (-1/3) ^ (-2) = -10 := by
  sorry

end calculation_proof_l2562_256296


namespace simplify_fraction_l2562_256213

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) :
  (m - 1) / m / ((m - 1) / (m^2)) = m := by
  sorry

end simplify_fraction_l2562_256213


namespace triangle_area_l2562_256234

-- Define the lines
def line1 (x y : ℝ) : Prop := y - 2*x = 3
def line2 (x y : ℝ) : Prop := 2*y - x = 9

-- Define the triangle
def triangle := {(x, y) : ℝ × ℝ | x ≥ 0 ∧ line1 x y ∧ line2 x y}

-- State the theorem
theorem triangle_area : MeasureTheory.volume triangle = 3/4 := by sorry

end triangle_area_l2562_256234


namespace passengers_taken_proof_l2562_256282

/-- The number of trains per hour -/
def trains_per_hour : ℕ := 12

/-- The number of passengers each train leaves at the station -/
def passengers_left_per_train : ℕ := 200

/-- The total number of passengers stepping on and off in an hour -/
def total_passengers_per_hour : ℕ := 6240

/-- The number of passengers each train takes from the station -/
def passengers_taken_per_train : ℕ := 320

theorem passengers_taken_proof :
  passengers_taken_per_train * trains_per_hour + 
  passengers_left_per_train * trains_per_hour = 
  total_passengers_per_hour :=
by sorry

end passengers_taken_proof_l2562_256282


namespace number_of_pencils_l2562_256274

theorem number_of_pencils (pens pencils : ℕ) : 
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 6 →
  pencils = 36 := by
sorry

end number_of_pencils_l2562_256274


namespace rhombus_side_length_l2562_256222

theorem rhombus_side_length 
  (d1 d2 : ℝ) 
  (h1 : d1 * d2 = 22) 
  (h2 : d1 + d2 = 10) 
  (h3 : (1/2) * d1 * d2 = 11) : 
  ∃ (side : ℝ), side = Real.sqrt 14 ∧ side^2 = (1/4) * (d1^2 + d2^2) := by
sorry

end rhombus_side_length_l2562_256222


namespace reservoir_fullness_after_storm_l2562_256206

theorem reservoir_fullness_after_storm 
  (original_content : ℝ) 
  (initial_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : original_content = 220)
  (h2 : initial_percentage = 55.00000000000001)
  (h3 : added_water = 120) : 
  (original_content + added_water) / (original_content / (initial_percentage / 100)) * 100 = 85 := by
sorry

end reservoir_fullness_after_storm_l2562_256206


namespace andrew_payment_l2562_256253

/-- The total amount Andrew paid for grapes and mangoes -/
def total_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1055 for his purchase -/
theorem andrew_payment : total_paid 8 70 9 55 = 1055 := by
  sorry

end andrew_payment_l2562_256253


namespace total_rectangles_is_176_l2562_256225

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of blue cells (more frequent gray cells) -/
def blue_cells : ℕ := 36

/-- The number of red cells (less frequent gray cells) -/
def red_cells : ℕ := 4

/-- The number of rectangles containing each blue cell -/
def rectangles_per_blue_cell : ℕ := 4

/-- The number of rectangles containing each red cell -/
def rectangles_per_red_cell : ℕ := 8

/-- The total number of rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue_cell + red_cells * rectangles_per_red_cell

theorem total_rectangles_is_176 : total_rectangles = 176 := by
  sorry

end total_rectangles_is_176_l2562_256225


namespace special_line_properties_l2562_256226

/-- A line passing through (-2, 3) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

theorem special_line_properties :
  (special_line (-2) 3) ∧
  (∃ a : ℝ, a ≠ 0 ∧ special_line (2 * a) 0 ∧ special_line 0 a) :=
by sorry

end special_line_properties_l2562_256226


namespace derivative_independent_of_function_value_l2562_256281

variable (f : ℝ → ℝ)
variable (x₀ : ℝ)

theorem derivative_independent_of_function_value :
  ∃ (g : ℝ → ℝ), g x₀ ≠ f x₀ ∧ HasDerivAt g (deriv f x₀) x₀ :=
sorry

end derivative_independent_of_function_value_l2562_256281


namespace tom_helicopter_rental_days_l2562_256285

/-- Calculates the number of days a helicopter was rented given the rental conditions and total payment -/
def helicopter_rental_days (hours_per_day : ℕ) (cost_per_hour : ℕ) (total_paid : ℕ) : ℕ :=
  total_paid / (hours_per_day * cost_per_hour)

/-- Theorem: Given Tom's helicopter rental conditions, he rented it for 3 days -/
theorem tom_helicopter_rental_days :
  helicopter_rental_days 2 75 450 = 3 := by
  sorry

end tom_helicopter_rental_days_l2562_256285


namespace negation_of_forall_proposition_l2562_256276

open Set

theorem negation_of_forall_proposition :
  (¬ ∀ x ∈ (Set.Ioo 0 1), x^2 - x < 0) ↔ (∃ x ∈ (Set.Ioo 0 1), x^2 - x ≥ 0) :=
by sorry

end negation_of_forall_proposition_l2562_256276


namespace quadratic_symmetry_l2562_256252

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem quadratic_symmetry (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  (m = 2 ∧
   (∀ x y, x < y → f m x < f m y) ∧
   (∀ x, x > 0 → f m x > f m 0) ∧
   (f m 0 = 2 ∧ ∀ x, f m x ≥ 2)) :=
by sorry

end quadratic_symmetry_l2562_256252


namespace function_values_l2562_256257

/-- A function from ℝ² to ℝ² defined by f(x, y) = (kx, y + b) -/
def f (k b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (k * x, y + b)

/-- Theorem stating that if f(3, 1) = (6, 2), then k = 2 and b = 1 -/
theorem function_values (k b : ℝ) : f k b (3, 1) = (6, 2) → k = 2 ∧ b = 1 := by
  sorry

end function_values_l2562_256257


namespace standard_deviation_constant_addition_original_sd_equals_new_sd_l2562_256209

/-- The standard deviation of a list of real numbers -/
noncomputable def standardDeviation (l : List ℝ) : ℝ := sorry

/-- Adding a constant to each element in a list -/
def addConstant (l : List ℝ) (c : ℝ) : List ℝ := sorry

theorem standard_deviation_constant_addition 
  (original : List ℝ) (c : ℝ) :
  standardDeviation original = standardDeviation (addConstant original c) :=
sorry

theorem original_sd_equals_new_sd 
  (original : List ℝ) (c : ℝ) :
  standardDeviation original = 2 → 
  standardDeviation (addConstant original c) = 2 :=
sorry

end standard_deviation_constant_addition_original_sd_equals_new_sd_l2562_256209


namespace at_most_one_right_or_obtuse_angle_l2562_256237

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  -- Sum of angles in a triangle is 180 degrees
  sum_180 : angle1 + angle2 + angle3 = 180

-- Theorem: At most one angle in a triangle is greater than or equal to 90 degrees
theorem at_most_one_right_or_obtuse_angle (t : Triangle) :
  (t.angle1 ≥ 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90) ∨
  (t.angle1 < 90 ∧ t.angle2 ≥ 90 ∧ t.angle3 < 90) ∨
  (t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 ≥ 90) :=
by
  sorry


end at_most_one_right_or_obtuse_angle_l2562_256237


namespace right_triangle_shorter_leg_l2562_256244

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- The shorter leg is 25 units
by sorry

end right_triangle_shorter_leg_l2562_256244


namespace min_recolor_is_n_minus_one_l2562_256242

/-- A complete graph of order n (≥ 3) with edges colored using three colors. -/
structure ColoredCompleteGraph where
  n : ℕ
  n_ge_3 : n ≥ 3
  colors : Fin 3 → Type
  edge_coloring : Fin n → Fin n → Fin 3
  each_color_used : ∀ c : Fin 3, ∃ i j : Fin n, i ≠ j ∧ edge_coloring i j = c

/-- The minimum number of edges that need to be recolored to make the graph connected by one color. -/
def min_recolor (G : ColoredCompleteGraph) : ℕ := G.n - 1

/-- Theorem stating that the minimum number of edges to recolor is n - 1. -/
theorem min_recolor_is_n_minus_one (G : ColoredCompleteGraph) :
  min_recolor G = G.n - 1 := by sorry

end min_recolor_is_n_minus_one_l2562_256242


namespace intersection_of_sets_l2562_256201

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 1, 2, 4}
  let B : Set ℤ := {-1, 0, 2}
  A ∩ B = {-1, 2} := by
sorry

end intersection_of_sets_l2562_256201


namespace birds_storks_difference_l2562_256264

theorem birds_storks_difference (initial_storks initial_birds additional_birds : ℕ) : 
  initial_storks = 5 →
  initial_birds = 3 →
  additional_birds = 4 →
  (initial_birds + additional_birds) - initial_storks = 2 :=
by sorry

end birds_storks_difference_l2562_256264


namespace math_competition_theorem_l2562_256200

/-- Represents the number of participants who solved both problem i and problem j -/
def p (i j : Fin 6) (n : ℕ) : ℕ := sorry

/-- Represents the number of participants who solved exactly k problems -/
def n_k (k : Fin 7) (n : ℕ) : ℕ := sorry

theorem math_competition_theorem (n : ℕ) :
  (∀ i j : Fin 6, i < j → p i j n > (2 * n) / 5) →
  (n_k 6 n = 0) →
  (n_k 5 n ≥ 2) :=
sorry

end math_competition_theorem_l2562_256200


namespace vector_problem_l2562_256269

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Parallel vectors in R^2 -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), v 0 * w 1 = t * v 1 * w 0

theorem vector_problem :
  (∃ k : ℝ, parallel (fun i => a i + k * c i) (fun i => 2 * b i + c i) → k = -11/18) ∧
  (∃ m n : ℝ, (∀ i, a i = m * b i - n * c i) → m = 5/9 ∧ n = -8/9) :=
sorry

end vector_problem_l2562_256269


namespace min_additional_marbles_for_john_l2562_256287

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem min_additional_marbles_for_john : 
  min_additional_marbles 15 60 = 60 := by
  sorry

end min_additional_marbles_for_john_l2562_256287


namespace magician_trick_l2562_256251

def is_valid_selection (a d : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ 16 ∧
  2 ≤ d ∧ d ≤ 16 ∧
  a % 2 = 0 ∧ d % 2 = 0 ∧
  a ≠ d ∧
  (d - a) % 16 = 3 ∨ (a - d) % 16 = 3

theorem magician_trick :
  ∃ (a d : ℕ), is_valid_selection a d ∧ a * d = 120 :=
sorry

end magician_trick_l2562_256251


namespace f_one_equals_one_l2562_256248

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_one_equals_one
  (f : ℝ → ℝ)
  (h : is_odd_function (fun x ↦ f (x + 1) - 1)) :
  f 1 = 1 := by
  sorry

end f_one_equals_one_l2562_256248


namespace horner_operations_for_f_l2562_256256

/-- The number of operations for Horner's method on a polynomial of degree n -/
def horner_operations (n : ℕ) : ℕ := 2 * n

/-- The polynomial f(x) = 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x + 1 -/
def f (x : ℝ) : ℝ := 3*x^6 + 4*x^5 + 5*x^4 + 6*x^3 + 7*x^2 + 8*x + 1

/-- The degree of the polynomial f -/
def degree_f : ℕ := 6

theorem horner_operations_for_f :
  horner_operations degree_f = 12 :=
sorry

end horner_operations_for_f_l2562_256256


namespace product_first_two_terms_l2562_256291

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem product_first_two_terms (a₁ : ℝ) (d : ℝ) :
  arithmetic_sequence a₁ d 7 = 25 ∧ d = 3 →
  a₁ * (a₁ + d) = 70 := by
  sorry

end product_first_two_terms_l2562_256291


namespace six_people_non_adjacent_seating_l2562_256260

/-- The number of ways to seat n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat n people around a round table
    where two specific individuals are adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := (n - 1) * Nat.factorial (n - 2)

/-- The number of ways to seat 6 people around a round table
    where two specific individuals are not adjacent. -/
theorem six_people_non_adjacent_seating :
  roundTableArrangements 6 - adjacentArrangements 6 = 24 := by
  sorry

end six_people_non_adjacent_seating_l2562_256260


namespace cube_volume_ratio_l2562_256210

-- Define the edge lengths
def edge_length_1 : ℚ := 5
def edge_length_2 : ℚ := 24  -- 2 feet = 24 inches

-- Define the volumes
def volume_1 : ℚ := edge_length_1^3
def volume_2 : ℚ := edge_length_2^3

-- Theorem statement
theorem cube_volume_ratio :
  volume_1 / volume_2 = 125 / 13824 := by
  sorry

end cube_volume_ratio_l2562_256210


namespace hundredth_ring_squares_l2562_256203

/-- The number of unit squares in the nth ring around a 2x2 central square -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 100th ring contains 808 unit squares -/
theorem hundredth_ring_squares : ring_squares 100 = 808 := by
  sorry

end hundredth_ring_squares_l2562_256203


namespace tournament_matches_l2562_256272

/-- Represents the number of matches played by each student -/
structure MatchCounts where
  student1 : Nat
  student2 : Nat
  student3 : Nat
  student4 : Nat
  student5 : Nat
  student6 : Nat

/-- The total number of matches in a tournament with 6 players -/
def totalMatches : Nat := 15

theorem tournament_matches (mc : MatchCounts) : 
  mc.student1 = 5 → 
  mc.student2 = 4 → 
  mc.student3 = 3 → 
  mc.student4 = 2 → 
  mc.student5 = 1 → 
  mc.student1 + mc.student2 + mc.student3 + mc.student4 + mc.student5 + mc.student6 = 2 * totalMatches → 
  mc.student6 = 3 := by
  sorry

end tournament_matches_l2562_256272


namespace no_integer_divisible_by_289_l2562_256228

theorem no_integer_divisible_by_289 :
  ∀ a : ℤ, ¬(289 ∣ (a^2 - 3*a - 19)) := by
sorry

end no_integer_divisible_by_289_l2562_256228


namespace kermit_sleep_positions_l2562_256227

/-- Represents a position on the infinite square grid -/
structure Position :=
  (x : Int) (y : Int)

/-- The number of Joules Kermit starts with -/
def initial_energy : Nat := 100

/-- Calculates the number of unique positions Kermit can reach -/
def unique_positions (energy : Nat) : Nat :=
  (2 * energy + 1) * (2 * energy + 1)

/-- Theorem stating the number of unique positions Kermit can reach -/
theorem kermit_sleep_positions : 
  unique_positions initial_energy = 10201 := by
  sorry

end kermit_sleep_positions_l2562_256227


namespace william_road_time_l2562_256239

def departure_time : Nat := 7 * 60  -- 7:00 AM in minutes
def arrival_time : Nat := 20 * 60  -- 8:00 PM in minutes
def stop_durations : List Nat := [25, 10, 25]

def total_journey_time : Nat := arrival_time - departure_time
def total_stop_time : Nat := stop_durations.sum

theorem william_road_time :
  (total_journey_time - total_stop_time) / 60 = 12 := by sorry

end william_road_time_l2562_256239


namespace alcohol_mixture_percentage_l2562_256271

theorem alcohol_mixture_percentage (original_volume : ℝ) (water_added : ℝ) (final_percentage : ℝ) :
  original_volume = 11 →
  water_added = 3 →
  final_percentage = 33 →
  (final_percentage / 100) * (original_volume + water_added) = 
    (42 / 100) * original_volume :=
by sorry

end alcohol_mixture_percentage_l2562_256271


namespace mnp_product_l2562_256245

theorem mnp_product (a b x y : ℝ) (m n p : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
  ((a^m*x - a^n) * (a^p*y - a^3) = a^5*b^5) → 
  m * n * p = 2 := by sorry

end mnp_product_l2562_256245


namespace vegetable_ghee_weight_l2562_256202

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 700

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3280

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

theorem vegetable_ghee_weight : 
  (weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume) + 
  (weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume) = total_weight :=
by sorry

end vegetable_ghee_weight_l2562_256202


namespace collinear_sufficient_not_necessary_for_coplanar_l2562_256280

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Determines if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Determines if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- The main theorem stating that three collinear points out of four
    is sufficient but not necessary for four points to be coplanar -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  ∃ (p q r s : Point3D),
    (collinear p q r → coplanar p q r s) ∧
    (coplanar p q r s ∧ ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s)) := by
  sorry

end collinear_sufficient_not_necessary_for_coplanar_l2562_256280


namespace min_value_sum_fractions_equality_condition_l2562_256235

theorem min_value_sum_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z :=
by sorry

end min_value_sum_fractions_equality_condition_l2562_256235


namespace years_since_same_average_l2562_256278

/-- Represents a club with members and their ages -/
structure Club where
  members : Nat
  avgAge : ℝ

/-- Represents the replacement of a member in the club -/
structure Replacement where
  oldMemberAge : ℝ
  newMemberAge : ℝ

/-- Theorem: The number of years since the average age was the same
    is equal to the age difference between the replaced and new member -/
theorem years_since_same_average (c : Club) (r : Replacement) :
  c.members = 5 →
  r.oldMemberAge - r.newMemberAge = 15 →
  c.avgAge * c.members = (c.avgAge * c.members - r.oldMemberAge + r.newMemberAge) →
  (r.oldMemberAge - r.newMemberAge : ℝ) = (c.avgAge * c.members - (c.avgAge * c.members - r.oldMemberAge + r.newMemberAge)) / c.members :=
by
  sorry


end years_since_same_average_l2562_256278


namespace triangle_with_sides_4_6_5_l2562_256289

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_with_sides_4_6_5 :
  can_form_triangle 4 6 5 ∧
  ¬can_form_triangle 4 6 2 ∧
  ¬can_form_triangle 4 6 10 ∧
  ¬can_form_triangle 4 6 11 :=
sorry

end triangle_with_sides_4_6_5_l2562_256289


namespace car_meeting_speed_l2562_256258

/-- Proves that given the conditions of the problem, the speed of the second car must be 60 mph -/
theorem car_meeting_speed (total_distance : ℝ) (speed1 : ℝ) (start_time1 start_time2 : ℝ) (x : ℝ) : 
  total_distance = 600 →
  speed1 = 50 →
  start_time1 = 7 →
  start_time2 = 8 →
  (total_distance / 2) / speed1 + start_time1 = (total_distance / 2) / x + start_time2 →
  x = 60 :=
by
  sorry

end car_meeting_speed_l2562_256258


namespace exists_irrational_greater_than_neg_three_l2562_256254

theorem exists_irrational_greater_than_neg_three :
  ∃ x : ℝ, Irrational x ∧ x > -3 := by sorry

end exists_irrational_greater_than_neg_three_l2562_256254


namespace ab_sufficient_not_necessary_for_a_plus_b_l2562_256212

theorem ab_sufficient_not_necessary_for_a_plus_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ a b, a > 0 → b > 0 → a * b > 1 → a + b > 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b > 2 ∧ a * b ≤ 1) := by
sorry

end ab_sufficient_not_necessary_for_a_plus_b_l2562_256212


namespace workshop_allocation_valid_l2562_256263

/-- Represents the allocation of workers in a workshop producing bolts and nuts. -/
structure WorkerAllocation where
  bolt_workers : ℕ
  nut_workers : ℕ

/-- Checks if a given worker allocation satisfies the workshop conditions. -/
def is_valid_allocation (total_workers : ℕ) (bolts_per_worker : ℕ) (nuts_per_worker : ℕ) 
    (nuts_per_bolt : ℕ) (allocation : WorkerAllocation) : Prop :=
  allocation.bolt_workers + allocation.nut_workers = total_workers ∧
  2 * (bolts_per_worker * allocation.bolt_workers) = nuts_per_worker * allocation.nut_workers

/-- Theorem stating that the specific allocation of 40 bolt workers and 50 nut workers
    is a valid solution to the workshop problem. -/
theorem workshop_allocation_valid : 
  is_valid_allocation 90 15 24 2 ⟨40, 50⟩ := by
  sorry


end workshop_allocation_valid_l2562_256263


namespace not_always_meaningful_regression_l2562_256286

-- Define the variables and their properties
variable (x y : ℝ)
variable (scatter_points : Set (ℝ × ℝ))

-- Define the conditions
def are_correlated (x y : ℝ) : Prop := sorry

def roughly_linear_distribution (points : Set (ℝ × ℝ)) : Prop := sorry

def regression_equation_meaningful (points : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem not_always_meaningful_regression 
  (h1 : are_correlated x y)
  (h2 : roughly_linear_distribution scatter_points) :
  ¬ ∀ (data : Set (ℝ × ℝ)), regression_equation_meaningful data :=
sorry

end not_always_meaningful_regression_l2562_256286


namespace blueprint_to_actual_length_l2562_256219

/-- Represents the scale of a blueprint in feet per inch -/
def blueprint_scale : ℝ := 500

/-- Represents the length of a line segment on the blueprint in inches -/
def blueprint_length : ℝ := 6.5

/-- Represents the actual length in feet corresponding to the blueprint length -/
def actual_length : ℝ := blueprint_scale * blueprint_length

theorem blueprint_to_actual_length :
  actual_length = 3250 := by sorry

end blueprint_to_actual_length_l2562_256219


namespace tan_sum_pi_twelfths_l2562_256233

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 8 := by
  sorry

end tan_sum_pi_twelfths_l2562_256233


namespace tissues_left_l2562_256288

/-- The number of tissues in one box -/
def tissues_per_box : ℕ := 160

/-- The number of boxes bought -/
def boxes_bought : ℕ := 3

/-- The number of tissues used -/
def tissues_used : ℕ := 210

/-- Theorem: Given the conditions, prove that the number of tissues left is 270 -/
theorem tissues_left : 
  tissues_per_box * boxes_bought - tissues_used = 270 := by
  sorry

end tissues_left_l2562_256288


namespace trapezium_height_l2562_256299

theorem trapezium_height (a b h : ℝ) (area : ℝ) : 
  a = 20 → b = 18 → area = 209 → (1/2) * (a + b) * h = area → h = 11 := by
  sorry

end trapezium_height_l2562_256299


namespace alexander_shopping_cost_l2562_256261

/-- Calculates the total cost of Alexander's shopping trip -/
def shopping_cost (apple_count : ℕ) (apple_price : ℕ) (orange_count : ℕ) (orange_price : ℕ) : ℕ :=
  apple_count * apple_price + orange_count * orange_price

/-- Theorem: Alexander spends $9 on his shopping trip -/
theorem alexander_shopping_cost :
  shopping_cost 5 1 2 2 = 9 := by
  sorry


end alexander_shopping_cost_l2562_256261


namespace inequality_proof_l2562_256268

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l2562_256268


namespace geometric_sequence_eighth_term_l2562_256295

theorem geometric_sequence_eighth_term
  (a₁ a₅ : ℚ)
  (h₁ : a₁ = 2187)
  (h₅ : a₅ = 960)
  (h_geom : ∃ r : ℚ, r ≠ 0 ∧ a₅ = a₁ * r^4) :
  ∃ a₈ : ℚ, a₈ = 35651584 / 4782969 ∧ (∃ r : ℚ, r ≠ 0 ∧ a₈ = a₁ * r^7) :=
by
  sorry


end geometric_sequence_eighth_term_l2562_256295


namespace range_of_c_l2562_256232

def p (c : ℝ) : Prop := c^2 < c

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 < 0

theorem range_of_c (c : ℝ) (h1 : p c ∨ q c) (h2 : ¬(p c ∧ q c)) :
  c ∈ Set.Icc (1/2) 1 ∪ Set.Ioc (-1/2) 0 :=
sorry

end range_of_c_l2562_256232


namespace binary_linear_equation_ab_eq_one_l2562_256221

/-- A binary linear equation is an equation of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def IsBinaryLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

theorem binary_linear_equation_ab_eq_one (a b : ℝ) :
  IsBinaryLinearEquation (fun x y => x^(2*a) + y^(b-1)) →
  a * b = 1 := by
  sorry

end binary_linear_equation_ab_eq_one_l2562_256221


namespace torn_sheets_count_l2562_256266

/-- Represents a book with consecutively numbered pages, two per sheet. -/
structure Book where
  /-- The last page number in the book -/
  last_page : ℕ

/-- Represents a set of consecutively torn-out sheets from a book -/
structure TornSheets where
  /-- The first torn-out page number -/
  first_page : ℕ
  /-- The last torn-out page number -/
  last_page : ℕ

/-- Check if two numbers have the same digits -/
def same_digits (a b : ℕ) : Prop := sorry

/-- Calculate the number of sheets torn out -/
def sheets_torn_out (ts : TornSheets) : ℕ :=
  (ts.last_page - ts.first_page + 1) / 2

/-- Main theorem -/
theorem torn_sheets_count (b : Book) (ts : TornSheets) :
  ts.first_page = 185 →
  same_digits ts.first_page ts.last_page →
  Even ts.last_page →
  ts.last_page > ts.first_page →
  ts.last_page ≤ b.last_page →
  sheets_torn_out ts = 167 := by sorry

end torn_sheets_count_l2562_256266


namespace least_positive_integer_satisfying_inequality_l2562_256246

theorem least_positive_integer_satisfying_inequality : 
  ∀ n : ℕ+, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15 ↔ n ≥ 4 :=
by sorry

end least_positive_integer_satisfying_inequality_l2562_256246


namespace fraction_simplification_l2562_256297

theorem fraction_simplification : (2 : ℚ) / 462 + 29 / 42 = 107 / 154 := by
  sorry

end fraction_simplification_l2562_256297


namespace dividend_calculation_l2562_256215

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.20)
  (h4 : dividend_rate = 0.07) :
  let price_per_share := face_value * (1 + premium_rate)
  let num_shares := investment / price_per_share
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by sorry

end dividend_calculation_l2562_256215


namespace geometric_sequence_and_max_function_l2562_256220

/-- Given that real numbers a, b, c, and d form a geometric sequence, 
    and the function y = ln(x + 2) - x attains its maximum value of c when x = b, 
    prove that ad = -1 -/
theorem geometric_sequence_and_max_function (a b c d : ℝ) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →
  (∀ x : ℝ, Real.log (x + 2) - x ≤ c) →
  (Real.log (b + 2) - b = c) →
  a * d = -1 := by
sorry

end geometric_sequence_and_max_function_l2562_256220


namespace banana_arrangements_l2562_256230

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let repeated_letter1 : ℕ := 3  -- 'a' appears 3 times
  let repeated_letter2 : ℕ := 2  -- 'n' appears 2 times
  factorial total_letters / (factorial repeated_letter1 * factorial repeated_letter2) = 60 := by
  sorry

end banana_arrangements_l2562_256230


namespace coefficient_of_x_squared_l2562_256204

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def binomial_expansion_coefficient (r : ℕ) : ℚ :=
  (-3)^r * binomial_coefficient 5 r

theorem coefficient_of_x_squared (expansion : ℕ → ℚ) :
  expansion = binomial_expansion_coefficient →
  (∃ r : ℕ, (10 - 3 * r) / 2 = 2 ∧ expansion r = 90) :=
sorry

end coefficient_of_x_squared_l2562_256204


namespace hot_dog_cost_l2562_256207

/-- Given that 6 hot dogs cost 300 cents in total, and each hot dog costs the same amount,
    prove that each hot dog costs 50 cents. -/
theorem hot_dog_cost (total_cost : ℕ) (num_hot_dogs : ℕ) (cost_per_hot_dog : ℕ) 
    (h1 : total_cost = 300)
    (h2 : num_hot_dogs = 6)
    (h3 : total_cost = num_hot_dogs * cost_per_hot_dog) : 
  cost_per_hot_dog = 50 := by
  sorry

end hot_dog_cost_l2562_256207


namespace optimal_oil_storage_l2562_256208

/-- Represents the optimal solution for storing oil in barrels -/
structure OilStorage where
  small_barrels : ℕ
  large_barrels : ℕ

/-- Checks if a given oil storage solution is valid -/
def is_valid_solution (total_oil : ℕ) (small_capacity : ℕ) (large_capacity : ℕ) (solution : OilStorage) : Prop :=
  solution.small_barrels * small_capacity + solution.large_barrels * large_capacity = total_oil

/-- Checks if a given oil storage solution is optimal -/
def is_optimal_solution (total_oil : ℕ) (small_capacity : ℕ) (large_capacity : ℕ) (solution : OilStorage) : Prop :=
  is_valid_solution total_oil small_capacity large_capacity solution ∧
  ∀ (other : OilStorage), 
    is_valid_solution total_oil small_capacity large_capacity other → 
    solution.small_barrels + solution.large_barrels ≤ other.small_barrels + other.large_barrels

/-- Theorem stating that the given solution is optimal for the oil storage problem -/
theorem optimal_oil_storage :
  is_optimal_solution 95 5 6 ⟨1, 15⟩ := by sorry

end optimal_oil_storage_l2562_256208


namespace roots_satisfy_conditions_l2562_256236

theorem roots_satisfy_conditions : ∃ (x y : ℝ),
  x + y = 10 ∧
  |x - y| = 12 ∧
  x^2 - 10*x - 22 = 0 ∧
  y^2 - 10*y - 22 = 0 := by
  sorry

end roots_satisfy_conditions_l2562_256236


namespace triangle_inequality_l2562_256277

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_inequality (t : Triangle) 
  (h1 : t.a^2 = t.b * (t.b + t.c))  -- Given condition
  (h2 : t.C > Real.pi / 2)          -- Angle C is obtuse
  : t.a < 2 * t.b ∧ 2 * t.b < t.c := by
  sorry

end triangle_inequality_l2562_256277


namespace ellipse_properties_l2562_256229

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the semi-focal distance
def semi_focal_distance : ℝ := 1

-- Define the condition that a > b > 0
def size_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0

-- Define the condition that the circle with diameter F₁F₂ passes through upper and lower vertices
def circle_condition (a b : ℝ) : Prop :=
  2 * semi_focal_distance = a

-- Theorem statement
theorem ellipse_properties (a b : ℝ) 
  (h1 : size_condition a b) 
  (h2 : circle_condition a b) :
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ k : ℝ, -Real.sqrt 2 / 2 < k ∧ k < 0 ∧
    ∀ x y : ℝ, y = k * (x - 2) → ellipse a b x y → y > 0 → x = 2 ∧ y = 0) :=
by sorry

end ellipse_properties_l2562_256229


namespace expansion_nonzero_terms_l2562_256216

/-- The number of nonzero terms in the expansion of (x^2+5)(3x^3+2x^2+6)-4(x^4-3x^3+8x^2+1) + 2x^3 -/
theorem expansion_nonzero_terms (x : ℝ) : 
  let expanded := (x^2 + 5) * (3*x^3 + 2*x^2 + 6) - 4*(x^4 - 3*x^3 + 8*x^2 + 1) + 2*x^3
  ∃ (a b c d e : ℝ) (n : ℕ), 
    expanded = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    n = 5 := by
  sorry

end expansion_nonzero_terms_l2562_256216


namespace fraction_inequality_l2562_256238

theorem fraction_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hab : a < b) (hcd : c < d) : 
  (a + c) / (b + c) < (a + d) / (b + d) := by
  sorry

end fraction_inequality_l2562_256238


namespace power_fraction_equality_l2562_256290

theorem power_fraction_equality : (16^6 * 8^3) / 4^10 = 2^13 := by
  sorry

end power_fraction_equality_l2562_256290


namespace unique_solution_l2562_256249

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 * y - x * y^2 - 5*x + 5*y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5*x^2 + 5*y^2 + 15 = 0

/-- The solution to the system of equations is unique and equal to (4, 1) -/
theorem unique_solution : ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (4, 1) := by sorry

end unique_solution_l2562_256249


namespace sams_morning_run_distance_l2562_256273

/-- Represents the distances traveled by Sam during different activities --/
structure SamDistances where
  morning_run : ℝ
  afternoon_walk : ℝ
  evening_bike : ℝ

/-- Theorem stating that given the conditions, Sam's morning run was 2 miles --/
theorem sams_morning_run_distance 
  (total_distance : ℝ) 
  (h1 : total_distance = 18) 
  (h2 : SamDistances → ℝ) 
  (h3 : ∀ d : SamDistances, h2 d = d.morning_run + d.afternoon_walk + d.evening_bike) 
  (h4 : ∀ d : SamDistances, d.afternoon_walk = 2 * d.morning_run) 
  (h5 : ∀ d : SamDistances, d.evening_bike = 12) :
  ∃ d : SamDistances, d.morning_run = 2 ∧ h2 d = total_distance := by
  sorry


end sams_morning_run_distance_l2562_256273


namespace dummies_leftover_l2562_256265

theorem dummies_leftover (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := by
  sorry

end dummies_leftover_l2562_256265


namespace fifth_hexagon_dots_l2562_256247

/-- The number of dots on each side of a hexagon layer -/
def dots_per_side (n : ℕ) : ℕ := n + 2

/-- The total number of dots in a single layer of a hexagon -/
def dots_in_layer (n : ℕ) : ℕ := 6 * (dots_per_side n)

/-- The total number of dots in a hexagon with n layers -/
def total_dots (n : ℕ) : ℕ := 
  if n = 0 then 0
  else total_dots (n - 1) + dots_in_layer n

/-- The fifth hexagon has 150 dots -/
theorem fifth_hexagon_dots : total_dots 5 = 150 := by
  sorry


end fifth_hexagon_dots_l2562_256247


namespace jiahao_estimate_l2562_256255

theorem jiahao_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x + 2) - (y - 1) > x - y := by
  sorry

end jiahao_estimate_l2562_256255


namespace rocketry_club_theorem_l2562_256294

theorem rocketry_club_theorem (total_students : ℕ) 
  (nails_neq_bolts : ℕ) (screws_eq_nails : ℕ) :
  total_students = 40 →
  nails_neq_bolts = 15 →
  screws_eq_nails = 10 →
  ∃ (screws_neq_bolts : ℕ), screws_neq_bolts ≥ 15 ∧
    screws_neq_bolts ≤ total_students - screws_eq_nails :=
by sorry

end rocketry_club_theorem_l2562_256294


namespace sphere_volume_larger_than_cube_l2562_256262

/-- Given a sphere and a cube with equal surface areas, the volume of the sphere is larger than the volume of the cube. -/
theorem sphere_volume_larger_than_cube (r : ℝ) (s : ℝ) (h : 4 * Real.pi * r^2 = 6 * s^2) :
  (4/3) * Real.pi * r^3 > s^3 := by
  sorry

end sphere_volume_larger_than_cube_l2562_256262


namespace remainder_theorem_l2562_256218

theorem remainder_theorem (s : ℤ) : 
  (s^15 - 2) % (s - 3) = 14348905 := by
  sorry

end remainder_theorem_l2562_256218


namespace sqrt_square_of_negative_l2562_256275

theorem sqrt_square_of_negative : Real.sqrt ((-2023)^2) = 2023 := by
  sorry

end sqrt_square_of_negative_l2562_256275


namespace completing_square_quadratic_l2562_256283

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by sorry

end completing_square_quadratic_l2562_256283


namespace radical_equation_condition_l2562_256292

theorem radical_equation_condition (x y : ℝ) : 
  xy ≠ 0 → (Real.sqrt (4 * x^2 * y^3) = -2 * x * y * Real.sqrt y ↔ x < 0 ∧ y > 0) :=
by sorry

end radical_equation_condition_l2562_256292


namespace distance_PF_is_five_l2562_256214

/-- Parabola structure with focus and directrix -/
structure Parabola :=
  (focus : ℝ × ℝ)
  (directrix : ℝ)

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) :=
  (point : ℝ × ℝ)
  (on_parabola : (point.2)^2 = 4 * point.1)

/-- Given parabola y^2 = 4x -/
def given_parabola : Parabola :=
  { focus := (1, 0),
    directrix := -1 }

/-- Point P on the parabola with x-coordinate 4 -/
def point_P : PointOnParabola given_parabola :=
  { point := (4, 4),
    on_parabola := by sorry }

/-- Theorem: The distance between P and F is 5 -/
theorem distance_PF_is_five :
  let F := given_parabola.focus
  let P := point_P.point
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5 := by sorry

end distance_PF_is_five_l2562_256214


namespace root_ordering_l2562_256270

/-- Given a quadratic function f(x) = (x-m)(x-n) + 2 where m < n,
    and α, β are the roots of f(x) = 0 with α < β,
    prove that m < α < β < n -/
theorem root_ordering (m n α β : ℝ) (hm : m < n) (hα : α < β)
  (hf : ∀ x, (x - m) * (x - n) + 2 = 0 ↔ x = α ∨ x = β) :
  m < α ∧ α < β ∧ β < n :=
sorry

end root_ordering_l2562_256270


namespace complex_sum_theorem_l2562_256259

theorem complex_sum_theorem (a b : ℂ) (h1 : a = 3 + 2*I) (h2 : b = 2 - I) :
  3*a + 4*b = 17 + 2*I := by
  sorry

end complex_sum_theorem_l2562_256259


namespace vote_ratio_l2562_256211

/-- Given a total of 60 votes and Ben receiving 24 votes, 
    prove that the ratio of votes received by Ben to votes received by Matt is 2:3 -/
theorem vote_ratio (total_votes : Nat) (ben_votes : Nat) 
    (h1 : total_votes = 60) 
    (h2 : ben_votes = 24) : 
  ∃ (matt_votes : Nat), 
    matt_votes = total_votes - ben_votes ∧ 
    (ben_votes : ℚ) / (matt_votes : ℚ) = 2 / 3 := by
  sorry

end vote_ratio_l2562_256211


namespace positive_number_equality_l2562_256293

theorem positive_number_equality (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (64/216) * (1/x)) : x = (2/9) * Real.sqrt 3 := by
  sorry

end positive_number_equality_l2562_256293


namespace product_of_sums_equals_3280_l2562_256298

theorem product_of_sums_equals_3280 :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_of_sums_equals_3280_l2562_256298


namespace investment_interest_l2562_256250

/-- Calculates the interest earned on an investment with compound interest -/
def interestEarned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Proves that the interest earned on a $5000 investment at 3% annual interest
    compounded annually for 10 years is $1720 (rounded to the nearest dollar) -/
theorem investment_interest : 
  Int.floor (interestEarned 5000 0.03 10) = 1720 := by
  sorry

end investment_interest_l2562_256250


namespace compound_composition_l2562_256223

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  hydrogen : ℕ
  chromium : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (comp : CompoundComposition) (h_weight o_weight cr_weight : ℚ) : ℚ :=
  comp.hydrogen * h_weight + comp.chromium * cr_weight + comp.oxygen * o_weight

/-- States that the compound has the given composition and molecular weight -/
theorem compound_composition (h_weight o_weight cr_weight : ℚ) :
  ∃ (comp : CompoundComposition),
    comp.chromium = 1 ∧
    comp.oxygen = 4 ∧
    molecularWeight comp h_weight o_weight cr_weight = 118 ∧
    h_weight = 1 ∧
    o_weight = 16 ∧
    cr_weight = 52 ∧
    comp.hydrogen = 2 := by
  sorry

end compound_composition_l2562_256223
