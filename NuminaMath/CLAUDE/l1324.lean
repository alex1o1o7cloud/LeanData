import Mathlib

namespace NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l1324_132404

theorem convex_quadrilaterals_from_circle_points (n : ℕ) (h : n = 20) :
  Nat.choose n 4 = 4845 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l1324_132404


namespace NUMINAMATH_CALUDE_language_school_solution_l1324_132485

/-- Represents the state of the language school at a given time --/
structure SchoolState where
  num_teachers : ℕ
  total_age : ℕ

/-- The language school problem --/
def language_school_problem (initial : SchoolState) 
  (new_teacher_age : ℕ) (left_teacher_age : ℕ) : Prop :=
  -- Initial state (2007)
  initial.num_teachers = 7 ∧
  -- State after new teacher joins (2010)
  (initial.total_age + 21 + new_teacher_age) / 8 = initial.total_age / 7 ∧
  -- State after one teacher leaves (2012)
  (initial.total_age + 37 + new_teacher_age - left_teacher_age) / 7 = initial.total_age / 7 ∧
  -- New teacher's age in 2010
  new_teacher_age = 25

theorem language_school_solution (initial : SchoolState) 
  (new_teacher_age : ℕ) (left_teacher_age : ℕ) 
  (h : language_school_problem initial new_teacher_age left_teacher_age) :
  left_teacher_age = 62 ∧ initial.total_age / 7 = 46 := by
  sorry

#check language_school_solution

end NUMINAMATH_CALUDE_language_school_solution_l1324_132485


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1324_132460

/-- Given that M(3,7) is the midpoint of CD and C(5,3) is one endpoint, 
    the product of the coordinates of point D is 11. -/
theorem midpoint_coordinate_product : 
  ∀ (D : ℝ × ℝ),
  (3, 7) = ((5 + D.1) / 2, (3 + D.2) / 2) →
  D.1 * D.2 = 11 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1324_132460


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1324_132434

theorem fixed_point_exponential_function 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 1
  f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1324_132434


namespace NUMINAMATH_CALUDE_nikolai_faster_l1324_132444

/-- Represents a mountain goat with a specific jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- The race parameters -/
def turning_point : ℕ := 2000

/-- Gennady's characteristics -/
def gennady : Goat := ⟨"Gennady", 6⟩

/-- Nikolai's characteristics -/
def nikolai : Goat := ⟨"Nikolai", 4⟩

/-- Calculates the number of jumps needed to reach the turning point -/
def jumps_to_turning_point (g : Goat) : ℕ :=
  (turning_point + g.jump_distance - 1) / g.jump_distance

/-- Calculates the total distance traveled to the turning point -/
def distance_to_turning_point (g : Goat) : ℕ :=
  (jumps_to_turning_point g) * g.jump_distance

/-- Theorem stating that Nikolai completes the journey faster -/
theorem nikolai_faster : 
  distance_to_turning_point nikolai < distance_to_turning_point gennady :=
sorry

end NUMINAMATH_CALUDE_nikolai_faster_l1324_132444


namespace NUMINAMATH_CALUDE_apples_left_l1324_132463

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Nancy picked -/
def nancy_apples : ℝ := 3.0

/-- The number of apples Keith ate -/
def keith_apples : ℝ := 6.0

/-- Theorem: The number of apples left after Mike and Nancy picked apples and Keith ate some -/
theorem apples_left : mike_apples + nancy_apples - keith_apples = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l1324_132463


namespace NUMINAMATH_CALUDE_missing_number_value_l1324_132447

theorem missing_number_value : 
  ∃ (x : ℚ), ((476 + 424) * 2 - x * 476 * 424 = 2704) ∧ (x = -1/223) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_value_l1324_132447


namespace NUMINAMATH_CALUDE_symmetric_point_to_origin_l1324_132496

/-- Given a point M with coordinates (-3, -5), proves that the coordinates of the point symmetric to M with respect to the origin are (3, 5). -/
theorem symmetric_point_to_origin (M : ℝ × ℝ) (h : M = (-3, -5)) :
  (- M.1, - M.2) = (3, 5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_to_origin_l1324_132496


namespace NUMINAMATH_CALUDE_approximately_200_men_joined_l1324_132465

-- Define the initial number of men
def initial_men : ℕ := 1000

-- Define the initial duration of provisions in days
def initial_duration : ℚ := 20

-- Define the new duration of provisions in days
def new_duration : ℚ := 167/10  -- 16.67 as a rational number

-- Define a function to calculate the number of men who joined
def men_joined : ℚ := 
  (initial_men * initial_duration / new_duration) - initial_men

-- Theorem statement
theorem approximately_200_men_joined : 
  199 ≤ men_joined ∧ men_joined < 201 := by
  sorry


end NUMINAMATH_CALUDE_approximately_200_men_joined_l1324_132465


namespace NUMINAMATH_CALUDE_rotate_180_of_A_l1324_132417

/-- Rotate a point 180 degrees about the origin -/
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-3, 2)

theorem rotate_180_of_A :
  rotate_180 A = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_180_of_A_l1324_132417


namespace NUMINAMATH_CALUDE_cards_given_to_friends_l1324_132458

theorem cards_given_to_friends (initial_cards : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 13 → remaining_cards = 4 → initial_cards - remaining_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_friends_l1324_132458


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1324_132484

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1324_132484


namespace NUMINAMATH_CALUDE_x_value_proof_l1324_132469

theorem x_value_proof (x : ℝ) : 
  3.5 * ((x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 → x = 3.6 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l1324_132469


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1324_132495

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  (∃ x y : ℝ, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1324_132495


namespace NUMINAMATH_CALUDE_meshed_gears_angular_velocity_ratio_l1324_132455

structure Gear where
  teeth : ℕ
  angularVelocity : ℝ

/-- The ratio of angular velocities for three meshed gears is proportional to the product of the other two gears' teeth counts. -/
theorem meshed_gears_angular_velocity_ratio 
  (A B C : Gear) 
  (h_mesh : A.angularVelocity * A.teeth = B.angularVelocity * B.teeth ∧ 
            B.angularVelocity * B.teeth = C.angularVelocity * C.teeth) :
  A.angularVelocity / (B.teeth * C.teeth) = 
  B.angularVelocity / (A.teeth * C.teeth) ∧
  B.angularVelocity / (A.teeth * C.teeth) = 
  C.angularVelocity / (A.teeth * B.teeth) :=
by sorry

end NUMINAMATH_CALUDE_meshed_gears_angular_velocity_ratio_l1324_132455


namespace NUMINAMATH_CALUDE_average_visitors_per_day_l1324_132401

def visitor_counts : List ℕ := [583, 246, 735, 492, 639]
def num_days : ℕ := 5

theorem average_visitors_per_day :
  (visitor_counts.sum / num_days : ℚ) = 539 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_per_day_l1324_132401


namespace NUMINAMATH_CALUDE_salamander_population_decline_l1324_132471

def decrease_rate : ℝ := 0.3
def target_percentage : ℝ := 0.05
def start_year : ℕ := 2007

def population_percentage (n : ℕ) : ℝ := (1 - decrease_rate) ^ n

theorem salamander_population_decline :
  ∃ n : ℕ, 
    population_percentage n ≤ target_percentage ∧
    population_percentage (n - 1) > target_percentage ∧
    start_year + n = 2016 :=
  sorry

end NUMINAMATH_CALUDE_salamander_population_decline_l1324_132471


namespace NUMINAMATH_CALUDE_trig_identity_l1324_132451

theorem trig_identity (α : ℝ) : 
  4.3 * Real.sin (4 * α) - Real.sin (5 * α) - Real.sin (6 * α) + Real.sin (7 * α) = 
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (11 * α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1324_132451


namespace NUMINAMATH_CALUDE_egg_distribution_l1324_132445

def crate_capacity : ℕ := 18
def abigail_eggs : ℕ := 58
def beatrice_eggs : ℕ := 76
def carson_eggs : ℕ := 27

def total_eggs : ℕ := abigail_eggs + beatrice_eggs + carson_eggs
def full_crates : ℕ := total_eggs / crate_capacity
def remaining_eggs : ℕ := total_eggs % crate_capacity

theorem egg_distribution :
  (remaining_eggs / 3 = 5) ∧
  (remaining_eggs % 3 = 2) ∧
  (abigail_eggs + 6 + beatrice_eggs + 6 + carson_eggs + 5 = total_eggs - full_crates * crate_capacity) := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l1324_132445


namespace NUMINAMATH_CALUDE_first_ring_hexagons_fiftieth_ring_hexagons_nth_ring_hexagons_l1324_132474

/-- The number of hexagons in the nth ring around a central hexagon in a hexagonal tiling -/
def hexagons_in_nth_ring (n : ℕ) : ℕ := 6 * n

/-- The first ring contains 6 hexagons -/
theorem first_ring_hexagons : hexagons_in_nth_ring 1 = 6 := by sorry

/-- The 50th ring contains 300 hexagons -/
theorem fiftieth_ring_hexagons : hexagons_in_nth_ring 50 = 300 := by sorry

/-- For any natural number n, the nth ring contains 6n hexagons -/
theorem nth_ring_hexagons (n : ℕ) : hexagons_in_nth_ring n = 6 * n := by sorry

end NUMINAMATH_CALUDE_first_ring_hexagons_fiftieth_ring_hexagons_nth_ring_hexagons_l1324_132474


namespace NUMINAMATH_CALUDE_graveyard_bones_problem_l1324_132431

theorem graveyard_bones_problem :
  let total_skeletons : ℕ := 20
  let adult_women : ℕ := total_skeletons / 2
  let adult_men : ℕ := (total_skeletons - adult_women) / 2
  let children : ℕ := total_skeletons - adult_women - adult_men
  let total_bones : ℕ := 375
  let woman_bones : ℕ → ℕ := λ x => x
  let man_bones : ℕ → ℕ := λ x => x + 5
  let child_bones : ℕ → ℕ := λ x => x / 2

  ∃ (w : ℕ), 
    adult_women * (woman_bones w) + 
    adult_men * (man_bones w) + 
    children * (child_bones w) = total_bones ∧ 
    w = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_graveyard_bones_problem_l1324_132431


namespace NUMINAMATH_CALUDE_randys_trip_distance_l1324_132414

theorem randys_trip_distance :
  ∀ y : ℝ,
  (y / 4 : ℝ) + 30 + (y / 3 : ℝ) = y →
  y = 72 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_distance_l1324_132414


namespace NUMINAMATH_CALUDE_system_solution_l1324_132456

theorem system_solution (x y : ℝ) : 
  x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97 ↔ 
  ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = -3 ∧ y = -2) ∨ (x = -2 ∧ y = -3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1324_132456


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1324_132408

/-- Represents an infinite geometric series -/
structure GeometricSeries where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum : ℝ -- sum of the series

/-- Condition for the first, third, and fourth terms forming an arithmetic sequence -/
def arithmeticSequenceCondition (s : GeometricSeries) : Prop :=
  2 * s.a * s.r^2 = s.a + s.a * s.r^3

/-- The main theorem statement -/
theorem geometric_series_first_term 
  (s : GeometricSeries) 
  (h_sum : s.sum = 2020)
  (h_arith : arithmeticSequenceCondition s)
  (h_converge : abs s.r < 1) :
  s.a = 1010 * (1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1324_132408


namespace NUMINAMATH_CALUDE_smaller_cube_edge_length_l1324_132430

/-- Given a cube with edge length 7 cm that is cut into smaller cubes, 
    if the total surface area increases by 600%, 
    then the edge length of the smaller cubes is 1 cm. -/
theorem smaller_cube_edge_length 
  (original_edge : ℝ) 
  (surface_area_increase : ℝ) 
  (smaller_edge : ℝ) : 
  original_edge = 7 →
  surface_area_increase = 6 →
  (6 * smaller_edge^2) * ((original_edge^3) / smaller_edge^3) = 
    (1 + surface_area_increase) * (6 * original_edge^2) →
  smaller_edge = 1 := by
sorry

end NUMINAMATH_CALUDE_smaller_cube_edge_length_l1324_132430


namespace NUMINAMATH_CALUDE_sum_equation_l1324_132419

theorem sum_equation (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y) : 
  2 * x + 3 * y + z = 20 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_l1324_132419


namespace NUMINAMATH_CALUDE_business_profit_calculation_l1324_132494

def business_profit (a_investment b_investment total_profit : ℚ) : ℚ :=
  let total_investment := a_investment + b_investment
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let a_share_ratio := a_investment / total_investment
  let a_share := a_share_ratio * remaining_profit
  management_fee + a_share

theorem business_profit_calculation :
  business_profit 3500 1500 9600 = 7008 :=
by sorry

end NUMINAMATH_CALUDE_business_profit_calculation_l1324_132494


namespace NUMINAMATH_CALUDE_no_square_from_square_cut_l1324_132476

-- Define a square
def Square (s : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

-- Define a straight cut
def StraightCut (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem: It's impossible to create a square from a larger square by a single straight cut
theorem no_square_from_square_cut (s₁ s₂ : ℝ) (h₁ : 0 < s₁) (h₂ : 0 < s₂) (h₃ : s₂ < s₁) :
  ¬∃ (a b c : ℝ), (Square s₁ ∩ StraightCut a b c).Nonempty ∧ 
    (Square s₂).Subset (Square s₁ ∩ StraightCut a b c) :=
sorry

end NUMINAMATH_CALUDE_no_square_from_square_cut_l1324_132476


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1324_132435

/-- Distance between foci of an ellipse -/
theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1324_132435


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l1324_132462

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) (n : ℕ) :
  a * (b * (c * n)) = (a * b * c) * n :=
by sorry

theorem problem_solution : (2 / 5 : ℚ) * ((3 / 4 : ℚ) * ((1 / 6 : ℚ) * 120)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l1324_132462


namespace NUMINAMATH_CALUDE_locus_and_max_area_l1324_132425

noncomputable section

-- Define the points E, F, and D
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (0, -2)

-- Define the moving point P
def P : ℝ × ℝ → Prop
  | (x, y) => (x + 2) * x + (y - 0) * y = 0 ∧ (x - 2) * x + (y - 0) * y = 0

-- Define the point M
def M : ℝ × ℝ → Prop
  | (x, y) => ∃ (px py : ℝ), P (px, py) ∧ px = x ∧ py = 2 * y

-- Define the locus C
def C : ℝ × ℝ → Prop
  | (x, y) => M (x, y)

-- Define the line l
def l (k : ℝ) : ℝ × ℝ → Prop
  | (x, y) => y = k * x - 2

-- Define the area of quadrilateral OANB
def area_OANB (k : ℝ) : ℝ := 
  8 * Real.sqrt ((4 * k^2 - 3) / (1 + 4 * k^2)^2)

-- Theorem statement
theorem locus_and_max_area :
  (∀ x y, C (x, y) ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ k₁ k₂, k₁ ≠ k₂ ∧
    area_OANB k₁ = 2 ∧
    area_OANB k₂ = 2 ∧
    (∀ k, area_OANB k ≤ 2) ∧
    l k₁ = λ (x, y) => y = Real.sqrt 7 / 2 * x - 2 ∧
    l k₂ = λ (x, y) => y = -Real.sqrt 7 / 2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_locus_and_max_area_l1324_132425


namespace NUMINAMATH_CALUDE_train_length_l1324_132438

/-- The length of a train given its speed and time to pass an observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 6 → speed_kmh * (1000 / 3600) * time_s = 240 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1324_132438


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1324_132420

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {y | -1 < y ∧ y < 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ Bᶜ = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1324_132420


namespace NUMINAMATH_CALUDE_positive_solution_x_l1324_132403

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 10 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 2 * z)
  (eq3 : x * z = 40 - 5 * x - 3 * z)
  (x_pos : x > 0) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l1324_132403


namespace NUMINAMATH_CALUDE_sugar_box_surface_area_l1324_132452

theorem sugar_box_surface_area :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →
    a * b * c = 280 →
    2 * (a * b + b * c + c * a) = 262 :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_box_surface_area_l1324_132452


namespace NUMINAMATH_CALUDE_cone_surface_area_l1324_132422

theorem cone_surface_area (r : ℝ) (h : r = 6) : 
  let sector_radius : ℝ := r
  let base_radius : ℝ := r / 2
  let slant_height : ℝ := sector_radius
  let base_area : ℝ := π * base_radius ^ 2
  let lateral_area : ℝ := π * base_radius * slant_height
  base_area + lateral_area = 27 * π := by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1324_132422


namespace NUMINAMATH_CALUDE_chen_recorded_steps_l1324_132454

/-- The standard number of steps for the walking activity -/
def standard : ℕ := 5000

/-- The function to calculate the recorded steps -/
def recorded_steps (actual_steps : ℕ) : ℤ :=
  (actual_steps : ℤ) - standard

/-- Theorem stating that 4800 actual steps should be recorded as -200 -/
theorem chen_recorded_steps :
  recorded_steps 4800 = -200 := by sorry

end NUMINAMATH_CALUDE_chen_recorded_steps_l1324_132454


namespace NUMINAMATH_CALUDE_sum_abc_equals_ten_l1324_132482

def f (x a b c : ℤ) : ℤ :=
  if x > 0 then a * x + b + 3
  else if x = 0 then a + b
  else 2 * b * x + c

theorem sum_abc_equals_ten (a b c : ℤ) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  f 2 a b c = 7 →
  f 0 a b c = 6 →
  f (-1) a b c = -4 →
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_ten_l1324_132482


namespace NUMINAMATH_CALUDE_union_of_sets_l1324_132489

theorem union_of_sets (a : ℤ) : 
  let A : Set ℤ := {|a + 1|, 3, 5}
  let B : Set ℤ := {2*a + 1, a^2 + 2*a, a^2 + 2*a - 1}
  (A ∩ B = {2, 3}) → (A ∪ B = {-5, 2, 3, 5}) :=
by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1324_132489


namespace NUMINAMATH_CALUDE_shopping_trip_result_l1324_132499

def shopping_trip (initial_amount : ℝ) (video_game_price : ℝ) (video_game_discount : ℝ)
  (goggles_percent : ℝ) (goggles_tax : ℝ) (jacket_price : ℝ) (jacket_discount : ℝ)
  (book_percent : ℝ) (book_tax : ℝ) (gift_card : ℝ) : ℝ :=
  let video_game_cost := video_game_price * (1 - video_game_discount)
  let remaining_after_game := initial_amount - video_game_cost
  let goggles_cost := remaining_after_game * goggles_percent * (1 + goggles_tax)
  let remaining_after_goggles := remaining_after_game - goggles_cost
  let jacket_cost := jacket_price * (1 - jacket_discount)
  let remaining_after_jacket := remaining_after_goggles - jacket_cost
  let book_cost := remaining_after_jacket * book_percent * (1 + book_tax)
  remaining_after_jacket - book_cost

theorem shopping_trip_result :
  shopping_trip 200 60 0.15 0.20 0.08 80 0.25 0.10 0.05 20 = 50.85 := by
  sorry

#eval shopping_trip 200 60 0.15 0.20 0.08 80 0.25 0.10 0.05 20

end NUMINAMATH_CALUDE_shopping_trip_result_l1324_132499


namespace NUMINAMATH_CALUDE_tom_shirt_purchase_l1324_132439

def shirts_per_fandom : ℕ := 5
def num_fandoms : ℕ := 4
def original_price : ℚ := 15
def discount_percentage : ℚ := 20
def tax_percentage : ℚ := 10

def discounted_price : ℚ := original_price * (1 - discount_percentage / 100)

def total_shirts : ℕ := shirts_per_fandom * num_fandoms

def pre_tax_total : ℚ := (total_shirts : ℚ) * discounted_price

def tax_amount : ℚ := pre_tax_total * (tax_percentage / 100)

def total_cost : ℚ := pre_tax_total + tax_amount

theorem tom_shirt_purchase :
  total_cost = 264 := by sorry

end NUMINAMATH_CALUDE_tom_shirt_purchase_l1324_132439


namespace NUMINAMATH_CALUDE_inverse_function_domain_l1324_132405

-- Define the function f(x) = -x(x+2)
def f (x : ℝ) : ℝ := -x * (x + 2)

-- State the theorem
theorem inverse_function_domain :
  {y : ℝ | ∃ x ≥ 0, f x = y} = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_inverse_function_domain_l1324_132405


namespace NUMINAMATH_CALUDE_meatball_fraction_eaten_l1324_132459

/-- Given 3 plates with 3 meatballs each, if 3 people eat the same fraction of meatballs from their respective plates and 3 meatballs are left in total, then each person ate 2/3 of the meatballs on their plate. -/
theorem meatball_fraction_eaten (f : ℚ) 
  (h1 : f ≥ 0) 
  (h2 : f ≤ 1) 
  (h3 : 3 * (3 - 3 * f) = 3) : 
  f = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_meatball_fraction_eaten_l1324_132459


namespace NUMINAMATH_CALUDE_prob_not_edge_10x10_l1324_132488

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  perimeter_squares : ℕ

/-- Calculates the probability of a randomly chosen square not touching the outer edge -/
def prob_not_edge (board : Checkerboard) : ℚ :=
  (board.total_squares - board.perimeter_squares : ℚ) / board.total_squares

/-- Theorem: The probability of a randomly chosen square not touching the outer edge
    on a 10x10 checkerboard is 16/25 -/
theorem prob_not_edge_10x10 :
  ∃ (board : Checkerboard),
    board.size = 10 ∧
    board.total_squares = 100 ∧
    board.perimeter_squares = 36 ∧
    prob_not_edge board = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_edge_10x10_l1324_132488


namespace NUMINAMATH_CALUDE_smallest_third_term_of_gp_l1324_132440

theorem smallest_third_term_of_gp (a b c : ℝ) : 
  (∃ d : ℝ, a = 9 ∧ b = 9 + d ∧ c = 9 + 2*d) →  -- arithmetic progression
  (∃ r : ℝ, 9 * (c + 20) = (b + 2)^2) →  -- geometric progression after modification
  (∃ x : ℝ, x ≥ c + 20 ∧ 
    ∀ y : ℝ, (∃ d : ℝ, 9 = 9 ∧ 9 + d + 2 = (9 * (9 + 2*d + 20))^(1/2) ∧ 9 + 2*d + 20 = y) 
    → x ≤ y) →
  1 ≤ c + 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_gp_l1324_132440


namespace NUMINAMATH_CALUDE_incenter_in_triangular_prism_l1324_132470

structure TriangularPrism where
  A : Point
  B : Point
  C : Point
  D : Point

def orthogonal_projection (p : Point) (plane : Set Point) : Point :=
  sorry

def distance_to_face (p : Point) (face : Set Point) : ℝ :=
  sorry

def is_incenter (p : Point) (triangle : Set Point) : Prop :=
  sorry

theorem incenter_in_triangular_prism (prism : TriangularPrism) 
  (O : Point) 
  (h1 : O = orthogonal_projection prism.A {prism.B, prism.C, prism.D}) 
  (h2 : distance_to_face O {prism.B, prism.C, prism.D} = 
        distance_to_face O {prism.A, prism.B, prism.D} ∧
        distance_to_face O {prism.B, prism.C, prism.D} = 
        distance_to_face O {prism.A, prism.C, prism.D}) : 
  is_incenter O {prism.B, prism.C, prism.D} :=
sorry

end NUMINAMATH_CALUDE_incenter_in_triangular_prism_l1324_132470


namespace NUMINAMATH_CALUDE_smallest_non_square_units_digit_l1324_132433

def is_square_units_digit (d : ℕ) : Prop :=
  ∃ n : ℕ, n^2 % 10 = d

theorem smallest_non_square_units_digit :
  (∀ d < 2, is_square_units_digit d) ∧
  ¬(is_square_units_digit 2) ∧
  (∀ d ≥ 2, ¬(is_square_units_digit d) → d ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_square_units_digit_l1324_132433


namespace NUMINAMATH_CALUDE_jessica_test_score_l1324_132486

-- Define the given conditions
def initial_students : ℕ := 20
def initial_average : ℚ := 75
def new_students : ℕ := 21
def new_average : ℚ := 76

-- Define Jessica's score as a variable
def jessica_score : ℚ := sorry

-- Theorem to prove
theorem jessica_test_score : 
  (initial_students * initial_average + jessica_score) / new_students = new_average := by
  sorry

end NUMINAMATH_CALUDE_jessica_test_score_l1324_132486


namespace NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l1324_132491

theorem winter_olympics_volunteer_allocation :
  let n_volunteers : ℕ := 5
  let n_events : ℕ := 4
  let allocation_schemes : ℕ := (n_volunteers.choose 2) * n_events.factorial
  allocation_schemes = 240 :=
by sorry

end NUMINAMATH_CALUDE_winter_olympics_volunteer_allocation_l1324_132491


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l1324_132427

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) 
  (h1 : total_packages = 27) 
  (h2 : total_pieces = 486) : 
  total_pieces / total_packages = 18 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l1324_132427


namespace NUMINAMATH_CALUDE_alpha_value_l1324_132432

theorem alpha_value (α γ : ℂ) 
  (h1 : (α + γ).re > 0)
  (h2 : (Complex.I * (α - 3 * γ)).re > 0)
  (h3 : γ = 4 + 3 * Complex.I) :
  α = 10.5 + 0.5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_alpha_value_l1324_132432


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1324_132413

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- Conditions for the ages -/
def age_conditions (a : Ages) : Prop :=
  a.roy = a.julia + 6 ∧
  a.roy + 2 = 2 * (a.julia + 2) ∧
  (a.roy + 2) * (a.kelly + 2) = 108

/-- The theorem to be proved -/
theorem age_ratio_is_two_to_one (a : Ages) :
  age_conditions a →
  (a.roy - a.julia) / (a.roy - a.kelly) = 2 := by
  sorry

#check age_ratio_is_two_to_one

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1324_132413


namespace NUMINAMATH_CALUDE_power_multiplication_l1324_132464

theorem power_multiplication (a : ℝ) : -a^4 * a^3 = -a^7 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l1324_132464


namespace NUMINAMATH_CALUDE_linear_function_characterization_l1324_132461

/-- A linear function f satisfying f(f(x)) = 16x - 15 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧
  (∀ x, f (f x) = 16 * x - 15)

/-- The theorem stating that a linear function satisfying f(f(x)) = 16x - 15 
    must be either 4x - 3 or -4x + 5 -/
theorem linear_function_characterization (f : ℝ → ℝ) :
  LinearFunction f → 
  ((∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l1324_132461


namespace NUMINAMATH_CALUDE_division_problem_l1324_132436

theorem division_problem (n : ℕ) : 
  n % 23 = 19 ∧ n / 23 = 17 → (10 * n) / 23 + (10 * n) % 23 = 184 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1324_132436


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1324_132492

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c (x + 5) = 4 * x^2 + 9 * x + 2) →
  a + b + c = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1324_132492


namespace NUMINAMATH_CALUDE_wheel_radius_increase_l1324_132406

/-- Calculates the increase in wheel radius given the original and new odometer readings,
    and the original wheel radius. --/
theorem wheel_radius_increase (original_reading : ℝ) (new_reading : ℝ) (original_radius : ℝ) 
  (h1 : original_reading = 390)
  (h2 : new_reading = 380)
  (h3 : original_radius = 12)
  (h4 : original_reading > new_reading) :
  ∃ (increase : ℝ), 
    0.265 < increase ∧ increase < 0.275 ∧ 
    (2 * Real.pi * (original_radius + increase) * new_reading = 
     2 * Real.pi * original_radius * original_reading) :=
by sorry

end NUMINAMATH_CALUDE_wheel_radius_increase_l1324_132406


namespace NUMINAMATH_CALUDE_minimum_value_a_l1324_132498

theorem minimum_value_a (a : ℝ) : (∀ x₁ x₂ x₃ x₄ : ℝ, ∃ k₁ k₂ k₃ k₄ : ℤ,
  (x₁ - k₁ - (x₂ - k₂))^2 + (x₁ - k₁ - (x₃ - k₃))^2 + (x₁ - k₁ - (x₄ - k₄))^2 +
  (x₂ - k₂ - (x₃ - k₃))^2 + (x₂ - k₂ - (x₄ - k₄))^2 + (x₃ - k₃ - (x₄ - k₄))^2 ≤ a) →
  a ≥ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_a_l1324_132498


namespace NUMINAMATH_CALUDE_carlas_classroom_desks_full_l1324_132473

/-- Represents the classroom setup and attendance for Carla's sixth-grade class -/
structure Classroom where
  total_students : ℕ
  restroom_students : ℕ
  rows : ℕ
  desks_per_row : ℕ

/-- Calculates the fraction of desks that are full in the classroom -/
def fraction_of_desks_full (c : Classroom) : ℚ :=
  let absent_students := 3 * c.restroom_students - 1
  let students_in_classroom := c.total_students - absent_students - c.restroom_students
  let total_desks := c.rows * c.desks_per_row
  (students_in_classroom : ℚ) / (total_desks : ℚ)

/-- Theorem stating that the fraction of desks full in Carla's classroom is 2/3 -/
theorem carlas_classroom_desks_full :
  ∃ (c : Classroom), c.total_students = 23 ∧ c.restroom_students = 2 ∧ c.rows = 4 ∧ c.desks_per_row = 6 ∧
  fraction_of_desks_full c = 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_carlas_classroom_desks_full_l1324_132473


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1324_132412

/-- Given two lines in the real plane, determine if a specific value of a parameter is sufficient but not necessary for their parallelism. -/
theorem parallel_lines_condition (a : ℝ) : 
  (∃ (x y : ℝ), a * x + 2 * y - 1 = 0) →  -- l₁ exists
  (∃ (x y : ℝ), x + (a + 1) * y + 4 = 0) →  -- l₂ exists
  (a = 1 → (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    a * x₁ + 2 * y₁ - 1 = 0 → 
    x₂ + (a + 1) * y₂ + 4 = 0 → 
    (y₂ - y₁) * (1 - 0) = (x₂ - x₁) * (2 - (a + 1)))) ∧ 
  (∃ b : ℝ, b ≠ 1 ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), 
      b * x₁ + 2 * y₁ - 1 = 0 → 
      x₂ + (b + 1) * y₂ + 4 = 0 → 
      (y₂ - y₁) * (1 - 0) = (x₂ - x₁) * (2 - (b + 1)))) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1324_132412


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1324_132472

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the focus condition
def focus_condition (a b : ℝ) : Prop :=
  ∃ c : ℝ, c = 4 ∧ c^2 = a^2 + b^2

-- Define the perpendicular asymptotes condition
def perpendicular_asymptotes (a b : ℝ) : Prop :=
  a = b

-- Main theorem
theorem hyperbola_equation (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : focus_condition a b) 
  (h4 : perpendicular_asymptotes a b) :
  ∀ x y : ℝ, hyperbola a b x y ↔ hyperbola (2 * Real.sqrt 2) (2 * Real.sqrt 2) x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1324_132472


namespace NUMINAMATH_CALUDE_cab_driver_income_l1324_132442

theorem cab_driver_income (day1 day2 day3 day4 day5 : ℕ) 
  (h1 : day1 = 250)
  (h2 : day2 = 400)
  (h4 : day4 = 400)
  (h5 : day5 = 500)
  (h_avg : (day1 + day2 + day3 + day4 + day5) / 5 = 460) :
  day3 = 750 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1324_132442


namespace NUMINAMATH_CALUDE_inequality_equiv_interval_l1324_132448

theorem inequality_equiv_interval (x : ℝ) (h : x + 3 ≠ 0) :
  (x + 1) / (x + 3) ≤ 3 ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-3) :=
sorry

end NUMINAMATH_CALUDE_inequality_equiv_interval_l1324_132448


namespace NUMINAMATH_CALUDE_salary_comparison_l1324_132441

/-- Given salaries in ratio 1:2:3 and sum of B and C's salaries is 6000,
    prove C's salary is 200% more than A's -/
theorem salary_comparison (a b c : ℕ) : 
  a + b + c > 0 →
  b = 2 * a →
  c = 3 * a →
  b + c = 6000 →
  (c - a) * 100 / a = 200 := by
sorry

end NUMINAMATH_CALUDE_salary_comparison_l1324_132441


namespace NUMINAMATH_CALUDE_convenient_logistics_boxes_l1324_132453

/-- Represents the number of large boxes -/
def large_boxes : ℕ := 8

/-- Represents the number of small boxes -/
def small_boxes : ℕ := 21 - large_boxes

/-- The total number of bottles -/
def total_bottles : ℕ := 2000

/-- The capacity of a large box -/
def large_box_capacity : ℕ := 120

/-- The capacity of a small box -/
def small_box_capacity : ℕ := 80

/-- The total number of boxes -/
def total_boxes : ℕ := 21

theorem convenient_logistics_boxes :
  large_boxes * large_box_capacity + small_boxes * small_box_capacity = total_bottles ∧
  large_boxes + small_boxes = total_boxes :=
by sorry

end NUMINAMATH_CALUDE_convenient_logistics_boxes_l1324_132453


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1324_132450

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -2) :
  (1 + 1 / (a - 1)) / (2 * a / (a^2 - 1)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1324_132450


namespace NUMINAMATH_CALUDE_class_average_after_exclusion_l1324_132428

theorem class_average_after_exclusion 
  (total_students : ℕ) 
  (total_average : ℚ) 
  (excluded_students : ℕ) 
  (excluded_average : ℚ) : 
  total_students = 10 → 
  total_average = 70 → 
  excluded_students = 5 → 
  excluded_average = 50 → 
  let remaining_students := total_students - excluded_students
  let remaining_total := total_students * total_average - excluded_students * excluded_average
  remaining_total / remaining_students = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_after_exclusion_l1324_132428


namespace NUMINAMATH_CALUDE_census_objects_eq_population_l1324_132481

/-- The entirety of objects under investigation in a census -/
def census_objects : Type := Unit

/-- The term "population" in statistical context -/
def population : Type := Unit

/-- Theorem stating that census objects are equivalent to population -/
theorem census_objects_eq_population : census_objects ≃ population := sorry

end NUMINAMATH_CALUDE_census_objects_eq_population_l1324_132481


namespace NUMINAMATH_CALUDE_quadratic_root_square_l1324_132493

theorem quadratic_root_square (p : ℝ) : 
  (∃ a b : ℝ, a ≠ b ∧ 
   a^2 - p*a + p = 0 ∧ 
   b^2 - p*b + p = 0 ∧ 
   (a = b^2 ∨ b = a^2)) ↔ 
  (p = 2 + Real.sqrt 5 ∨ p = 2 - Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_l1324_132493


namespace NUMINAMATH_CALUDE_integral_tangent_sine_cosine_l1324_132426

open Real MeasureTheory

theorem integral_tangent_sine_cosine :
  ∫ x in (Set.Icc 0 (π/4)), (7 + 3 * tan x) / (sin x + 2 * cos x)^2 = 3 * log (3/2) + 1/6 := by
  sorry

end NUMINAMATH_CALUDE_integral_tangent_sine_cosine_l1324_132426


namespace NUMINAMATH_CALUDE_min_value_sin_function_l1324_132411

theorem min_value_sin_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x - π / 4)) :
  ∃ x ∈ Set.Icc 0 (π / 2), ∀ y ∈ Set.Icc 0 (π / 2), f x ≤ f y ∧ f x = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_function_l1324_132411


namespace NUMINAMATH_CALUDE_alice_original_seat_l1324_132437

/-- Represents the possible seats in the lecture hall -/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six
  | seven

/-- Represents the movement of a person -/
inductive Movement
  | left : Nat → Movement
  | right : Nat → Movement
  | stay : Movement
  | switch : Movement

/-- Represents a person and their movement -/
structure Person where
  name : String
  movement : Movement

/-- The state of the seating arrangement -/
structure SeatingArrangement where
  seats : Vector Person 7
  aliceOriginalSeat : Seat
  aliceFinalSeat : Seat

def isEndSeat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.seven

/-- The theorem to prove -/
theorem alice_original_seat
  (arrangement : SeatingArrangement)
  (beth_moves_right : arrangement.seats[1].movement = Movement.right 1)
  (carla_moves_left : arrangement.seats[2].movement = Movement.left 2)
  (dana_elly_switch : arrangement.seats[3].movement = Movement.switch ∧
                      arrangement.seats[4].movement = Movement.switch)
  (fiona_moves_left : arrangement.seats[5].movement = Movement.left 1)
  (grace_stays : arrangement.seats[6].movement = Movement.stay)
  (alice_ends_in_end_seat : isEndSeat arrangement.aliceFinalSeat) :
  arrangement.aliceOriginalSeat = Seat.five := by
  sorry

end NUMINAMATH_CALUDE_alice_original_seat_l1324_132437


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1324_132479

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 12*x + 4 = 0 ↔ (x + c)^2 = d ∧ d = 32 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1324_132479


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_in_pascal_l1324_132409

/-- Represents a row in Pascal's triangle -/
def PascalRow := List Nat

/-- Generates the next row of Pascal's triangle given the current row -/
def nextPascalRow (row : PascalRow) : PascalRow :=
  sorry

/-- Checks if a number is a four-digit number -/
def isFourDigit (n : Nat) : Bool :=
  1000 ≤ n ∧ n ≤ 9999

/-- Finds the nth four-digit number in Pascal's triangle -/
def nthFourDigitInPascal (n : Nat) : Nat :=
  sorry

theorem third_smallest_four_digit_in_pascal :
  nthFourDigitInPascal 3 = 1002 :=
sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_in_pascal_l1324_132409


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1324_132446

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 60

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1324_132446


namespace NUMINAMATH_CALUDE_library_visitors_average_l1324_132477

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (month_days : ℕ) (sundays_in_month : ℕ) :
  sunday_visitors = 510 →
  other_day_visitors = 240 →
  month_days = 30 →
  sundays_in_month = 5 →
  (sundays_in_month * sunday_visitors + (month_days - sundays_in_month) * other_day_visitors) / month_days = 285 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l1324_132477


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l1324_132490

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/6, 1/3 + 1/9]
  (∀ x ∈ sums, 1/3 + 1/9 ≤ x) ∧ (1/3 + 1/9 = 4/9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l1324_132490


namespace NUMINAMATH_CALUDE_equal_cupcake_distribution_l1324_132467

theorem equal_cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end NUMINAMATH_CALUDE_equal_cupcake_distribution_l1324_132467


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1324_132410

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (A < π) →
  (B > 0) → (B < π) →
  (C > 0) → (C < π) →
  (A + B + C = π) →
  ((Real.sqrt 3 * a) / (1 + Real.cos A) = c / Real.sin C) →
  (a = Real.sqrt 3) →
  (c - b = (Real.sqrt 6 - Real.sqrt 2) / 2) →
  (A = π / 3 ∧ (1/2 * b * c * Real.sin A = (3 + Real.sqrt 3) / 4)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1324_132410


namespace NUMINAMATH_CALUDE_equation_solution_l1324_132418

/-- Given the equation P = s / (1 + k + m)^n, prove that n = log(s/P) / log(1 + k + m) -/
theorem equation_solution (P s k m n : ℝ) (h : P = s / (1 + k + m)^n) :
  n = Real.log (s / P) / Real.log (1 + k + m) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1324_132418


namespace NUMINAMATH_CALUDE_zoe_correct_percentage_l1324_132424

theorem zoe_correct_percentage
  (total : ℝ)
  (chloe_alone : ℝ)
  (zoe_alone : ℝ)
  (amy_alone : ℝ)
  (together : ℝ)
  (chloe_correct_alone : ℝ)
  (chloe_correct_overall : ℝ)
  (zoe_correct_alone : ℝ)
  (trio_correct_together : ℝ)
  (h1 : chloe_alone = 0.4 * total)
  (h2 : zoe_alone = 0.3 * total)
  (h3 : amy_alone = 0.3 * total)
  (h4 : together = total - (chloe_alone + zoe_alone + amy_alone))
  (h5 : chloe_correct_alone = 0.8 * chloe_alone)
  (h6 : chloe_correct_overall = 0.88 * (chloe_alone + together))
  (h7 : zoe_correct_alone = 0.75 * zoe_alone)
  (h8 : trio_correct_together = 0.85 * together)
  : (zoe_correct_alone + trio_correct_together) / (zoe_alone + together) = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_zoe_correct_percentage_l1324_132424


namespace NUMINAMATH_CALUDE_range_of_k_l1324_132402

theorem range_of_k (n : ℕ+) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k ∧ |x₂ - 2*n| = k) →
  0 < k ∧ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l1324_132402


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1324_132423

/-- A parabola is defined by its equation y² = 2x -/
def Parabola := {(x, y) : ℝ × ℝ | y^2 = 2*x}

/-- The distance from the focus to the directrix of a parabola y² = 2x is 1 -/
theorem parabola_focus_directrix_distance :
  ∃ (f d : ℝ × ℝ), f ∈ Parabola ∧ (∀ (p : ℝ × ℝ), p ∈ Parabola → dist p d = dist p f) ∧ dist f d = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1324_132423


namespace NUMINAMATH_CALUDE_triangle_semicircle_inequality_l1324_132407

-- Define a triangle by its semiperimeter and inradius
structure Triangle where
  s : ℝ  -- semiperimeter
  r : ℝ  -- inradius
  s_pos : 0 < s
  r_pos : 0 < r

-- Define the radius of the circle tangent to the three semicircles
noncomputable def t (tri : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_semicircle_inequality (tri : Triangle) :
  tri.s / 2 < t tri ∧ t tri ≤ tri.s / 2 + (1 - Real.sqrt 3 / 2) * tri.r := by
  sorry

end NUMINAMATH_CALUDE_triangle_semicircle_inequality_l1324_132407


namespace NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l1324_132449

theorem max_value_3xy_plus_yz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  3*x*y + y*z ≤ Real.sqrt 10 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l1324_132449


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1324_132487

/-- Calculates the total cost of production given fixed cost, marginal cost, and number of products. -/
def totalCost (fixedCost marginalCost : ℕ) (numProducts : ℕ) : ℕ :=
  fixedCost + marginalCost * numProducts

/-- Proves that the total cost of producing 20 products is $16,000, given a fixed cost of $12,000 and a marginal cost of $200 per product. -/
theorem total_cost_calculation :
  totalCost 12000 200 20 = 16000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1324_132487


namespace NUMINAMATH_CALUDE_divisibility_condition_l1324_132466

theorem divisibility_condition (a : ℕ) : 
  (a^2 + a + 1) ∣ (a^7 + 3*a^6 + 3*a^5 + 3*a^4 + a^3 + a^2 + 3) ↔ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1324_132466


namespace NUMINAMATH_CALUDE_sons_age_l1324_132497

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 34 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 32 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1324_132497


namespace NUMINAMATH_CALUDE_distance_to_line_is_sqrt_17_l1324_132457

/-- The distance from a point to a line in 3D space --/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the distance from (2, 0, -1) to the line passing through (1, 3, 1) and (3, -1, 5) is √17 --/
theorem distance_to_line_is_sqrt_17 :
  distance_point_to_line (2, 0, -1) (1, 3, 1) (3, -1, 5) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_is_sqrt_17_l1324_132457


namespace NUMINAMATH_CALUDE_new_light_wattage_l1324_132416

theorem new_light_wattage (old_wattage : ℝ) (increase_percentage : ℝ) (new_wattage : ℝ) :
  old_wattage = 80 →
  increase_percentage = 0.25 →
  new_wattage = old_wattage * (1 + increase_percentage) →
  new_wattage = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_new_light_wattage_l1324_132416


namespace NUMINAMATH_CALUDE_gas_price_increase_l1324_132443

/-- Given two successive price increases in gas, where the second increase is 20%,
    and a driver needs to reduce gas consumption by 35.89743589743589% to keep
    expenditure constant, prove that the first price increase was approximately 30%. -/
theorem gas_price_increase (initial_price : ℝ) (initial_consumption : ℝ) :
  initial_price > 0 →
  initial_consumption > 0 →
  ∃ (first_increase : ℝ),
    (initial_price * initial_consumption =
      initial_price * (1 + first_increase / 100) * 1.20 * initial_consumption * (1 - 35.89743589743589 / 100)) ∧
    (abs (first_increase - 30) < 0.00001) := by
  sorry

end NUMINAMATH_CALUDE_gas_price_increase_l1324_132443


namespace NUMINAMATH_CALUDE_largest_argument_l1324_132475

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z - 10i| = 5√2
def satisfies_condition (z : ℂ) : Prop :=
  Complex.abs (z - Complex.I * 10) = 5 * Real.sqrt 2

-- Define the theorem
theorem largest_argument :
  ∃ (z : ℂ), satisfies_condition z ∧
  ∀ (w : ℂ), satisfies_condition w → Complex.arg w ≤ Complex.arg z ∧
  z = -5 + 5 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_largest_argument_l1324_132475


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l1324_132480

theorem sum_and_ratio_to_difference (x y : ℝ) 
  (sum_eq : x + y = 780) 
  (ratio_eq : x / y = 1.25) : 
  x - y = 86 + 2/3 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l1324_132480


namespace NUMINAMATH_CALUDE_distance_between_points_l1324_132415

/-- The distance between points (1, 2) and (5, 6) is 4√2 units. -/
theorem distance_between_points : Real.sqrt ((5 - 1)^2 + (6 - 2)^2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1324_132415


namespace NUMINAMATH_CALUDE_log_and_power_equality_l1324_132468

theorem log_and_power_equality : 
  (Real.log 32 - Real.log 4) / Real.log 2 + (27 : ℝ) ^ (2/3) = 12 := by sorry

end NUMINAMATH_CALUDE_log_and_power_equality_l1324_132468


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l1324_132421

theorem min_cubes_for_box (box_length box_width box_height cube_volume : ℕ) 
  (h1 : box_length = 10)
  (h2 : box_width = 13)
  (h3 : box_height = 5)
  (h4 : cube_volume = 5) :
  (box_length * box_width * box_height) / cube_volume = 130 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l1324_132421


namespace NUMINAMATH_CALUDE_football_yards_gained_l1324_132483

-- Define the initial loss
def initial_loss : Int := 5

-- Define the final progress
def final_progress : Int := 3

-- Theorem statement
theorem football_yards_gained :
  ∃ (yards_gained : Int), -initial_loss + yards_gained = final_progress ∧ yards_gained = 8 := by
sorry

end NUMINAMATH_CALUDE_football_yards_gained_l1324_132483


namespace NUMINAMATH_CALUDE_park_creatures_l1324_132400

theorem park_creatures (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300)
  (h2 : total_legs = 686) : ∃ (birds mammals imaginary : ℕ),
  birds + mammals + imaginary = total_heads ∧
  2 * birds + 4 * mammals + 3 * imaginary = total_legs ∧
  birds = 214 := by
  sorry

end NUMINAMATH_CALUDE_park_creatures_l1324_132400


namespace NUMINAMATH_CALUDE_equal_real_imag_parts_l1324_132429

theorem equal_real_imag_parts (b : ℝ) : 
  let z : ℂ := (1 + I) / (1 - I) + (1 / 2 : ℂ) * b
  (z.re = z.im) ↔ b = 2 := by sorry

end NUMINAMATH_CALUDE_equal_real_imag_parts_l1324_132429


namespace NUMINAMATH_CALUDE_ian_money_left_l1324_132478

/-- Calculates the amount of money Ian has left after paying off debts --/
def money_left (lottery_win : ℕ) (colin_payment : ℕ) : ℕ :=
  let helen_payment := 2 * colin_payment
  let benedict_payment := helen_payment / 2
  lottery_win - (colin_payment + helen_payment + benedict_payment)

/-- Theorem stating that Ian has $20 left after paying off debts --/
theorem ian_money_left : money_left 100 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ian_money_left_l1324_132478
