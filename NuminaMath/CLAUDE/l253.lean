import Mathlib

namespace NUMINAMATH_CALUDE_total_journey_time_l253_25394

/-- Represents the problem of Joe's journey to school -/
structure JourneyToSchool where
  d : ℝ  -- Total distance from home to school
  walk_speed : ℝ  -- Joe's walking speed
  run_speed : ℝ  -- Joe's running speed
  walk_time : ℝ  -- Time Joe takes to walk 1/3 of the distance

/-- Conditions of the problem -/
def journey_conditions (j : JourneyToSchool) : Prop :=
  j.run_speed = 4 * j.walk_speed ∧
  j.walk_time = 9 ∧
  j.walk_speed * j.walk_time = j.d / 3

/-- The theorem to be proved -/
theorem total_journey_time (j : JourneyToSchool) 
  (h : journey_conditions j) : 
  ∃ (total_time : ℝ), total_time = 13.5 ∧ 
    total_time = j.walk_time + (2 * j.d / 3) / j.run_speed :=
by sorry

end NUMINAMATH_CALUDE_total_journey_time_l253_25394


namespace NUMINAMATH_CALUDE_well_digging_payment_l253_25343

/-- The total amount paid to two workers for digging a well --/
def total_amount_paid (hours_day1 hours_day2 hours_day3 : ℕ) (hourly_rate : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := 2 * total_hours
  total_man_hours * hourly_rate

/-- Theorem stating that the total amount paid is $660 --/
theorem well_digging_payment :
  total_amount_paid 10 8 15 10 = 660 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l253_25343


namespace NUMINAMATH_CALUDE_image_of_two_zero_l253_25389

/-- A mapping that transforms a point (x, y) into (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The image of the point (2, 0) under the mapping f is (2, 2) -/
theorem image_of_two_zero :
  f (2, 0) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_zero_l253_25389


namespace NUMINAMATH_CALUDE_simplify_expression_l253_25309

theorem simplify_expression : (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l253_25309


namespace NUMINAMATH_CALUDE_smallest_box_volume_l253_25379

/-- Represents a triangular pyramid (tetrahedron) -/
structure Pyramid where
  height : ℝ
  base_side : ℝ

/-- Represents a rectangular prism -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def box_volume (b : Box) : ℝ :=
  b.length * b.width * b.height

/-- Checks if a box can safely contain a pyramid -/
def can_contain (b : Box) (p : Pyramid) : Prop :=
  b.height ≥ p.height ∧ b.length ≥ p.base_side ∧ b.width ≥ p.base_side

/-- The smallest box that can safely contain the pyramid -/
def smallest_box (p : Pyramid) : Box :=
  { length := 10, width := 10, height := p.height }

/-- Theorem: The volume of the smallest box that can safely contain the given pyramid is 1500 cubic inches -/
theorem smallest_box_volume (p : Pyramid) (h1 : p.height = 15) (h2 : p.base_side = 8) :
  box_volume (smallest_box p) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_box_volume_l253_25379


namespace NUMINAMATH_CALUDE_wendy_flowers_proof_l253_25329

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The number of flowers that wilted -/
def wilted_flowers : ℕ := 35

/-- The number of bouquets that can be made after some flowers wilted -/
def remaining_bouquets : ℕ := 2

/-- The initial number of flowers Wendy picked -/
def initial_flowers : ℕ := wilted_flowers + remaining_bouquets * flowers_per_bouquet

theorem wendy_flowers_proof : initial_flowers = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendy_flowers_proof_l253_25329


namespace NUMINAMATH_CALUDE_smallest_n_has_9_digits_l253_25393

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def has_9_digits (n : ℕ) : Prop := n ≥ 100000000 ∧ n < 1000000000

theorem smallest_n_has_9_digits :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(is_divisible_by m 30 ∧ is_perfect_cube (m^2) ∧ is_perfect_square (m^5))) ∧
    is_divisible_by n 30 ∧
    is_perfect_cube (n^2) ∧
    is_perfect_square (n^5) ∧
    has_9_digits n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_has_9_digits_l253_25393


namespace NUMINAMATH_CALUDE_function_composition_l253_25354

-- Define the function f
def f : ℝ → ℝ := fun x => 3 * (x + 1) - 1

-- State the theorem
theorem function_composition (x : ℝ) : f x = 3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l253_25354


namespace NUMINAMATH_CALUDE_can_obtain_all_graphs_l253_25319

/-- Represents a candidate in the election -/
structure Candidate where
  id : Nat

/-- Represents a voter's ranking of candidates -/
structure Ranking where
  preferences : List Candidate

/-- Represents the election system -/
structure ElectionSystem where
  candidates : Finset Candidate
  voters : Finset Nat
  rankings : Nat → Ranking

/-- Represents a directed graph -/
structure DirectedGraph where
  vertices : Finset Candidate
  edges : Candidate → Candidate → Bool

/-- Counts the number of votes where a is ranked higher than b -/
def countPreferences (system : ElectionSystem) (a b : Candidate) : Nat :=
  sorry

/-- Checks if there should be an edge from a to b based on majority preference -/
def hasEdge (system : ElectionSystem) (a b : Candidate) : Bool :=
  2 * countPreferences system a b > system.voters.card

/-- Constructs a directed graph based on the election system -/
def constructGraph (system : ElectionSystem) : DirectedGraph :=
  sorry

/-- Theorem stating that any connected complete directed graph can be obtained -/
theorem can_obtain_all_graphs (n : Nat) :
  ∃ (system : ElectionSystem),
    system.candidates.card = n ∧
    system.voters.card = n ∧
    ∀ (g : DirectedGraph),
      g.vertices = system.candidates →
      ∃ (newSystem : ElectionSystem),
        newSystem.candidates = system.candidates ∧
        constructGraph newSystem = g :=
  sorry

end NUMINAMATH_CALUDE_can_obtain_all_graphs_l253_25319


namespace NUMINAMATH_CALUDE_binary_representation_of_51_l253_25353

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNumber :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Theorem: The binary representation of 51 is 110011 -/
theorem binary_representation_of_51 :
  toBinary 51 = [true, true, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_51_l253_25353


namespace NUMINAMATH_CALUDE_oxford_high_school_population_is_1247_l253_25322

/-- Represents the number of people in Oxford High School -/
def oxford_high_school_population : ℕ :=
  let full_time_teachers : ℕ := 80
  let part_time_teachers : ℕ := 5
  let principal : ℕ := 1
  let vice_principals : ℕ := 3
  let librarians : ℕ := 2
  let guidance_counselors : ℕ := 6
  let other_staff : ℕ := 25
  let classes : ℕ := 40
  let avg_students_per_class : ℕ := 25
  let part_time_students : ℕ := 250

  let full_time_students : ℕ := classes * avg_students_per_class
  let total_staff : ℕ := full_time_teachers + part_time_teachers + principal + 
                         vice_principals + librarians + guidance_counselors + other_staff
  let total_students : ℕ := full_time_students + (part_time_students / 2)

  total_staff + total_students

/-- Theorem stating that the total number of people in Oxford High School is 1247 -/
theorem oxford_high_school_population_is_1247 : 
  oxford_high_school_population = 1247 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_population_is_1247_l253_25322


namespace NUMINAMATH_CALUDE_lauras_remaining_pay_l253_25351

/-- Calculates the remaining amount of Laura's pay after expenses --/
def remaining_pay (hourly_rate : ℚ) (hours_per_day : ℚ) (days_worked : ℚ) 
                  (food_clothing_percentage : ℚ) (rent : ℚ) : ℚ :=
  let total_earnings := hourly_rate * hours_per_day * days_worked
  let food_clothing_expense := total_earnings * food_clothing_percentage
  let remaining_after_food_clothing := total_earnings - food_clothing_expense
  remaining_after_food_clothing - rent

/-- Theorem stating that Laura's remaining pay is $250 --/
theorem lauras_remaining_pay :
  remaining_pay 10 8 10 (1/4) 350 = 250 := by
  sorry

end NUMINAMATH_CALUDE_lauras_remaining_pay_l253_25351


namespace NUMINAMATH_CALUDE_equation_positive_root_l253_25346

theorem equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x - a) / (x - 1) - 3 / x = 1) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_positive_root_l253_25346


namespace NUMINAMATH_CALUDE_mark_sprint_distance_l253_25335

/-- The distance traveled by Mark given his sprint time and speed -/
theorem mark_sprint_distance (time : ℝ) (speed : ℝ) (h1 : time = 24.0) (h2 : speed = 6.0) :
  time * speed = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_mark_sprint_distance_l253_25335


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l253_25303

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 + (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₁| + |a₂| + |a₅| = 105 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l253_25303


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l253_25376

theorem neither_sufficient_nor_necessary : ¬(∀ x : ℝ, -1 < x ∧ x < 2 → |x - 2| < 1) ∧
                                           ¬(∀ x : ℝ, |x - 2| < 1 → -1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l253_25376


namespace NUMINAMATH_CALUDE_min_vertical_distance_l253_25333

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 - 3*x - 5

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), vertical_distance x₀ ≤ vertical_distance x ∧ vertical_distance x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l253_25333


namespace NUMINAMATH_CALUDE_siblings_have_extra_money_l253_25310

def perfume_cost : ℚ := 100
def christian_savings : ℚ := 7
def sue_savings : ℚ := 9
def bob_savings : ℚ := 3
def christian_yards : ℕ := 7
def christian_yard_rate : ℚ := 7
def sue_dogs : ℕ := 10
def sue_dog_rate : ℚ := 4
def bob_families : ℕ := 5
def bob_family_rate : ℚ := 2
def discount_rate : ℚ := 20 / 100

def total_earnings : ℚ :=
  christian_savings + sue_savings + bob_savings +
  christian_yards * christian_yard_rate +
  sue_dogs * sue_dog_rate +
  bob_families * bob_family_rate

def discounted_price : ℚ :=
  perfume_cost * (1 - discount_rate)

theorem siblings_have_extra_money :
  total_earnings - discounted_price = 38 := by sorry

end NUMINAMATH_CALUDE_siblings_have_extra_money_l253_25310


namespace NUMINAMATH_CALUDE_new_girl_weight_l253_25344

/-- Given a group of 10 girls, if replacing one girl weighing 50 kg with a new girl
    increases the average weight by 5 kg, then the new girl weighs 100 kg. -/
theorem new_girl_weight (initial_weight : ℝ) (new_weight : ℝ) :
  (initial_weight - 50 + new_weight) / 10 = initial_weight / 10 + 5 →
  new_weight = 100 := by
sorry

end NUMINAMATH_CALUDE_new_girl_weight_l253_25344


namespace NUMINAMATH_CALUDE_line_intersects_circle_l253_25345

/-- Proves that a line intersects a circle given specific conditions -/
theorem line_intersects_circle (x₀ y₀ a : ℝ) (h1 : a > 0) (h2 : x₀^2 + y₀^2 > a^2) :
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ x₀*x + y₀*y = a^2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l253_25345


namespace NUMINAMATH_CALUDE_f_composition_seven_l253_25312

-- Define the function f
def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

-- State the theorem
theorem f_composition_seven : f (f (f (f (f (f 7))))) = 116 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_seven_l253_25312


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l253_25336

/-- The coefficient of x^2 in the expansion of (1 + 1/x)(1-x)^7 is -14 -/
theorem coefficient_x_squared_expansion : ℤ := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l253_25336


namespace NUMINAMATH_CALUDE_debby_hat_tickets_l253_25398

/-- The number of tickets Debby spent on various items at the arcade -/
structure ArcadeTickets where
  total : ℕ
  stuffedAnimal : ℕ
  yoyo : ℕ
  hat : ℕ

/-- Theorem stating that given the conditions, Debby spent 2 tickets on the hat -/
theorem debby_hat_tickets (tickets : ArcadeTickets) 
    (h1 : tickets.total = 14)
    (h2 : tickets.stuffedAnimal = 10)
    (h3 : tickets.yoyo = 2)
    (h4 : tickets.total = tickets.stuffedAnimal + tickets.yoyo + tickets.hat) : 
  tickets.hat = 2 := by
  sorry

end NUMINAMATH_CALUDE_debby_hat_tickets_l253_25398


namespace NUMINAMATH_CALUDE_B_work_time_l253_25378

/-- The number of days it takes for B to complete the work alone -/
def days_for_B : ℝ := 20

/-- The fraction of work completed by A and B together in 2 days -/
def work_completed_in_2_days : ℝ := 1 - 0.7666666666666666

theorem B_work_time (days_for_A : ℝ) (h1 : days_for_A = 15) :
  2 * (1 / days_for_A + 1 / days_for_B) = work_completed_in_2_days := by
  sorry

#check B_work_time

end NUMINAMATH_CALUDE_B_work_time_l253_25378


namespace NUMINAMATH_CALUDE_product_expansion_l253_25372

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x) - 5 * x^3) = 3 / x - (15 / 7) * x^3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l253_25372


namespace NUMINAMATH_CALUDE_walk_ratio_l253_25308

def distance_first_hour : ℝ := 2
def total_distance : ℝ := 6

def distance_second_hour : ℝ := total_distance - distance_first_hour

theorem walk_ratio :
  distance_second_hour / distance_first_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_walk_ratio_l253_25308


namespace NUMINAMATH_CALUDE_initial_fish_count_l253_25342

def fish_eaten_per_day : ℕ := 2
def days_before_adding : ℕ := 14
def fish_added : ℕ := 8
def days_after_adding : ℕ := 7
def final_fish_count : ℕ := 26

theorem initial_fish_count (initial_count : ℕ) : 
  initial_count - (fish_eaten_per_day * days_before_adding) + fish_added - 
  (fish_eaten_per_day * days_after_adding) = final_fish_count → 
  initial_count = 60 := by
sorry

end NUMINAMATH_CALUDE_initial_fish_count_l253_25342


namespace NUMINAMATH_CALUDE_expression_value_l253_25383

theorem expression_value (b : ℚ) (h : b = 1/3) :
  (3 * b⁻¹ + b⁻¹ / 3) / b = 30 := by sorry

end NUMINAMATH_CALUDE_expression_value_l253_25383


namespace NUMINAMATH_CALUDE_inequality_relation_l253_25320

theorem inequality_relation (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by sorry

end NUMINAMATH_CALUDE_inequality_relation_l253_25320


namespace NUMINAMATH_CALUDE_rhombus_from_equal_triangle_perimeters_l253_25348

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- The perimeter of a triangle defined by three points -/
def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Theorem: If the perimeters of triangles ABO, BCO, CDO, and DAO are equal
    in a convex quadrilateral ABCD where O is the intersection of diagonals,
    then ABCD is a rhombus -/
theorem rhombus_from_equal_triangle_perimeters (q : Quadrilateral) 
  (h_convex : isConvex q) :
  let O := diagonalIntersection q
  (trianglePerimeter q.A q.B O = trianglePerimeter q.B q.C O) ∧
  (trianglePerimeter q.B q.C O = trianglePerimeter q.C q.D O) ∧
  (trianglePerimeter q.C q.D O = trianglePerimeter q.D q.A O) →
  (q.A.x - q.B.x)^2 + (q.A.y - q.B.y)^2 = 
  (q.B.x - q.C.x)^2 + (q.B.y - q.C.y)^2 ∧
  (q.B.x - q.C.x)^2 + (q.B.y - q.C.y)^2 = 
  (q.C.x - q.D.x)^2 + (q.C.y - q.D.y)^2 ∧
  (q.C.x - q.D.x)^2 + (q.C.y - q.D.y)^2 = 
  (q.D.x - q.A.x)^2 + (q.D.y - q.A.y)^2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_from_equal_triangle_perimeters_l253_25348


namespace NUMINAMATH_CALUDE_phantoms_initial_money_l253_25386

def black_ink_cost : ℕ := 11
def red_ink_cost : ℕ := 15
def yellow_ink_cost : ℕ := 13
def black_ink_quantity : ℕ := 2
def red_ink_quantity : ℕ := 3
def yellow_ink_quantity : ℕ := 2
def additional_amount_needed : ℕ := 43

theorem phantoms_initial_money :
  black_ink_quantity * black_ink_cost +
  red_ink_quantity * red_ink_cost +
  yellow_ink_quantity * yellow_ink_cost -
  additional_amount_needed = 50 := by
    sorry

end NUMINAMATH_CALUDE_phantoms_initial_money_l253_25386


namespace NUMINAMATH_CALUDE_smallest_c_for_quadratic_inequality_l253_25366

theorem smallest_c_for_quadratic_inequality : 
  ∃ c : ℝ, c = 2 ∧ (∀ x : ℝ, -x^2 + 9*x - 14 ≥ 0 → x ≥ c) := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_quadratic_inequality_l253_25366


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l253_25381

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l253_25381


namespace NUMINAMATH_CALUDE_baseball_players_l253_25347

/-- Given a club with the following properties:
  * There are 310 people in total
  * 138 people play tennis
  * 94 people play both tennis and baseball
  * 11 people do not play any sport
  Prove that 255 people play baseball -/
theorem baseball_players (total : ℕ) (tennis : ℕ) (both : ℕ) (none : ℕ) 
  (h1 : total = 310)
  (h2 : tennis = 138)
  (h3 : both = 94)
  (h4 : none = 11) :
  total - (tennis - both) - none = 255 := by
  sorry

#eval 310 - (138 - 94) - 11

end NUMINAMATH_CALUDE_baseball_players_l253_25347


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l253_25390

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.000000301 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.01 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l253_25390


namespace NUMINAMATH_CALUDE_g_neg_two_l253_25325

def g (x : ℝ) : ℝ := x^3 - 2*x + 1

theorem g_neg_two : g (-2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_l253_25325


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_verify_solution_l253_25313

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given a SwimmerSpeed and a direction. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 10 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed)
  (h_downstream : effectiveSpeed s true * 3 = 45)
  (h_upstream : effectiveSpeed s false * 3 = 15) : 
  s.swimmer = 10 := by
  sorry

/-- Verifies that the solution satisfies the given conditions. -/
theorem verify_solution : 
  let s : SwimmerSpeed := ⟨10, 5⟩
  effectiveSpeed s true * 3 = 45 ∧ 
  effectiveSpeed s false * 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_verify_solution_l253_25313


namespace NUMINAMATH_CALUDE_real_complex_intersection_l253_25362

-- Define the set of real numbers
def RealNumbers : Set ℂ := {z : ℂ | z.im = 0}

-- Define the set of complex numbers
def ComplexNumbers : Set ℂ := Set.univ

-- Theorem statement
theorem real_complex_intersection :
  RealNumbers ∩ ComplexNumbers = RealNumbers := by sorry

end NUMINAMATH_CALUDE_real_complex_intersection_l253_25362


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l253_25339

/-- Proves that the discount percentage is 15% given the conditions of the dress pricing problem -/
theorem dress_discount_percentage : ∀ (original_price : ℝ) (discount_percentage : ℝ),
  original_price > 0 →
  discount_percentage > 0 →
  discount_percentage < 100 →
  original_price * (1 - discount_percentage / 100) = 68 →
  68 * 1.25 = original_price - 5 →
  discount_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l253_25339


namespace NUMINAMATH_CALUDE_circles_tangent_m_value_l253_25355

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define external tangency condition
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m ∧
  ∀ (x' y' : ℝ), C₁ x' y' → C₂ x' y' m → (x = x' ∧ y = y')

-- Theorem statement
theorem circles_tangent_m_value :
  ∀ m : ℝ, externally_tangent m → m = 9 :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_m_value_l253_25355


namespace NUMINAMATH_CALUDE_student_number_problem_l253_25382

theorem student_number_problem (x y : ℤ) : 
  x = 121 → 2 * x - y = 102 → y = 140 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l253_25382


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l253_25331

/-- Given a point (2, 7, z) in 3D space, where z is unknown, and its distance 
    from the origin (0, 0, 0) is 10 units, prove that z = √47. -/
theorem fly_distance_from_ceiling :
  ∀ z : ℝ, (2:ℝ)^2 + 7^2 + z^2 = 10^2 → z = Real.sqrt 47 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l253_25331


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l253_25349

/-- Proves that given a car's average speed over two hours and its speed in the second hour, 
    we can determine its speed in the first hour. -/
theorem car_speed_first_hour 
  (average_speed : ℝ) 
  (second_hour_speed : ℝ) 
  (h1 : average_speed = 90) 
  (h2 : second_hour_speed = 60) : 
  ∃ (first_hour_speed : ℝ), 
    first_hour_speed = 120 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l253_25349


namespace NUMINAMATH_CALUDE_first_runner_time_l253_25357

/-- Represents a 600-meter relay race with three runners -/
structure RelayRace where
  runner1_time : ℝ
  runner2_time : ℝ
  runner3_time : ℝ

/-- The conditions of the specific relay race -/
def race_conditions (race : RelayRace) : Prop :=
  race.runner2_time = race.runner1_time + 2 ∧
  race.runner3_time = race.runner1_time - 3 ∧
  race.runner1_time + race.runner2_time + race.runner3_time = 71

/-- Theorem stating that given the race conditions, the first runner's time is 24 seconds -/
theorem first_runner_time (race : RelayRace) :
  race_conditions race → race.runner1_time = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_first_runner_time_l253_25357


namespace NUMINAMATH_CALUDE_smaller_hexagon_area_ratio_l253_25373

/-- A regular hexagon with side length 4 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 4)

/-- Midpoint of a side of the hexagon -/
structure Midpoint :=
  (point : ℝ × ℝ)

/-- The smaller hexagon formed by connecting midpoints of alternating sides -/
structure SmallerHexagon :=
  (vertices : List (ℝ × ℝ))
  (is_regular : Bool)

/-- The ratio of the area of the smaller hexagon to the area of the original hexagon -/
def area_ratio (original : RegularHexagon) (smaller : SmallerHexagon) : ℚ :=
  49/36

theorem smaller_hexagon_area_ratio 
  (original : RegularHexagon) 
  (G H I J K L : Midpoint) 
  (smaller : SmallerHexagon) :
  area_ratio original smaller = 49/36 :=
sorry

end NUMINAMATH_CALUDE_smaller_hexagon_area_ratio_l253_25373


namespace NUMINAMATH_CALUDE_ratio_of_powers_l253_25358

theorem ratio_of_powers : (2^17 * 3^19) / 6^18 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_powers_l253_25358


namespace NUMINAMATH_CALUDE_arianna_position_l253_25314

/-- The length of the race in meters -/
def race_length : ℝ := 1000

/-- The distance between Ethan and Arianna when Ethan finished, in meters -/
def distance_between : ℝ := 816

/-- Arianna's distance from the start line when Ethan finished -/
def arianna_distance : ℝ := race_length - distance_between

theorem arianna_position : arianna_distance = 184 := by
  sorry

end NUMINAMATH_CALUDE_arianna_position_l253_25314


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l253_25334

/-- The speed of the moon around the Earth in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.03

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the moon around the Earth in kilometers per hour -/
def moon_speed_km_per_hour : ℝ := moon_speed_km_per_sec * seconds_per_hour

theorem moon_speed_conversion :
  moon_speed_km_per_hour = 3708 := by sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l253_25334


namespace NUMINAMATH_CALUDE_final_price_fraction_l253_25367

/-- The final price of a dress for a staff member after discounts and tax -/
def final_price (d : ℝ) : ℝ :=
  let discount_price := d * (1 - 0.45)
  let staff_price := discount_price * (1 - 0.40)
  staff_price * (1 + 0.08)

/-- Theorem stating the final price as a fraction of the initial price -/
theorem final_price_fraction (d : ℝ) :
  final_price d = 0.3564 * d := by
  sorry

end NUMINAMATH_CALUDE_final_price_fraction_l253_25367


namespace NUMINAMATH_CALUDE_selection_schemes_with_women_l253_25364

/-- The number of ways to select 4 individuals from 4 men and 2 women, with at least 1 woman included -/
def selection_schemes (total_men : ℕ) (total_women : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose (total_men + total_women) to_select - Nat.choose total_men to_select

theorem selection_schemes_with_women (total_men : ℕ) (total_women : ℕ) (to_select : ℕ) 
    (h1 : total_men = 4)
    (h2 : total_women = 2)
    (h3 : to_select = 4) :
  selection_schemes total_men total_women to_select = 14 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_with_women_l253_25364


namespace NUMINAMATH_CALUDE_sparrow_population_decrease_l253_25399

/-- The annual decrease rate of the sparrow population -/
def decrease_rate : ℝ := 0.3

/-- The threshold percentage of the initial population -/
def threshold : ℝ := 0.2

/-- The remaining population fraction after one year -/
def remaining_fraction : ℝ := 1 - decrease_rate

/-- The number of years it takes for the population to fall below the threshold -/
def years_to_threshold : ℕ := 5

theorem sparrow_population_decrease :
  (remaining_fraction ^ years_to_threshold) < threshold ∧
  ∀ n : ℕ, n < years_to_threshold → (remaining_fraction ^ n) ≥ threshold :=
by sorry

end NUMINAMATH_CALUDE_sparrow_population_decrease_l253_25399


namespace NUMINAMATH_CALUDE_base_7_addition_sum_l253_25370

-- Define a function to convert a base 7 number to base 10
def to_base_10 (x : ℕ) (y : ℕ) (z : ℕ) : ℕ := x * 49 + y * 7 + z

-- Define the addition problem in base 7
def addition_problem (X Y : ℕ) : Prop :=
  to_base_10 2 X Y + to_base_10 0 5 2 = to_base_10 3 1 X

-- Define the condition that X and Y are single digits in base 7
def single_digit_base_7 (n : ℕ) : Prop := n < 7

theorem base_7_addition_sum :
  ∀ X Y : ℕ,
    addition_problem X Y →
    single_digit_base_7 X →
    single_digit_base_7 Y →
    X + Y = 4 :=
by sorry

end NUMINAMATH_CALUDE_base_7_addition_sum_l253_25370


namespace NUMINAMATH_CALUDE_right_handed_players_l253_25369

/-- The number of right-handed players on a cricket team -/
theorem right_handed_players (total : ℕ) (throwers : ℕ) : 
  total = 55 →
  throwers = 37 →
  throwers ≤ total →
  (total - throwers) % 3 = 0 →  -- Ensures one-third of non-throwers can be left-handed
  49 = throwers + (total - throwers) - (total - throwers) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_l253_25369


namespace NUMINAMATH_CALUDE_fraction_simplification_l253_25327

theorem fraction_simplification :
  (20 : ℚ) / 21 * 35 / 54 * 63 / 50 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l253_25327


namespace NUMINAMATH_CALUDE_calculate_expression_l253_25307

theorem calculate_expression : (-2 + 3) * 2 + (-2)^3 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l253_25307


namespace NUMINAMATH_CALUDE_initial_number_of_people_l253_25318

/-- Given a group of people where replacing one person increases the average weight,
    this theorem proves the initial number of people in the group. -/
theorem initial_number_of_people
  (n : ℕ) -- Initial number of people
  (weight_increase_per_person : ℝ) -- Average weight increase per person
  (weight_difference : ℝ) -- Weight difference between new and replaced person
  (h1 : weight_increase_per_person = 2.5)
  (h2 : weight_difference = 20)
  (h3 : n * weight_increase_per_person = weight_difference) :
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_people_l253_25318


namespace NUMINAMATH_CALUDE_school_trip_ratio_l253_25374

theorem school_trip_ratio (total : ℕ) (remaining : ℕ) : 
  total = 1000 → 
  remaining = 250 → 
  (total / 2 - remaining) / remaining = 1 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_ratio_l253_25374


namespace NUMINAMATH_CALUDE_bernie_postcard_transaction_l253_25360

theorem bernie_postcard_transaction (initial_postcards : ℕ) 
  (sell_price : ℕ) (buy_price : ℕ) : 
  initial_postcards = 18 → 
  sell_price = 15 → 
  buy_price = 5 → 
  (initial_postcards / 2 * sell_price) / buy_price = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_bernie_postcard_transaction_l253_25360


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l253_25306

theorem pet_shop_kittens (total : ℕ) (hamsters : ℕ) (birds : ℕ) (kittens : ℕ) : 
  total = 77 → hamsters = 15 → birds = 30 → kittens = total - hamsters - birds → kittens = 32 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_kittens_l253_25306


namespace NUMINAMATH_CALUDE_power_two_minus_one_div_by_seven_l253_25375

theorem power_two_minus_one_div_by_seven (n : ℕ) : 
  7 ∣ (2^n - 1) ↔ 3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_power_two_minus_one_div_by_seven_l253_25375


namespace NUMINAMATH_CALUDE_floor_width_calculation_l253_25316

/-- Given a rectangular floor of length 10 m, covered by a square carpet of side 4 m,
    with 64 square meters uncovered, the width of the floor is 8 m. -/
theorem floor_width_calculation (floor_length : ℝ) (carpet_side : ℝ) (uncovered_area : ℝ) :
  floor_length = 10 →
  carpet_side = 4 →
  uncovered_area = 64 →
  ∃ (width : ℝ), width = 8 ∧ floor_length * width = carpet_side^2 + uncovered_area :=
by sorry

end NUMINAMATH_CALUDE_floor_width_calculation_l253_25316


namespace NUMINAMATH_CALUDE_fourth_cd_cost_l253_25368

theorem fourth_cd_cost (initial_avg_cost : ℝ) (new_avg_cost : ℝ) (initial_cd_count : ℕ) :
  initial_avg_cost = 15 →
  new_avg_cost = 16 →
  initial_cd_count = 3 →
  (initial_cd_count * initial_avg_cost + (new_avg_cost * (initial_cd_count + 1) - initial_cd_count * initial_avg_cost)) = 19 := by
  sorry

end NUMINAMATH_CALUDE_fourth_cd_cost_l253_25368


namespace NUMINAMATH_CALUDE_number_puzzle_l253_25350

theorem number_puzzle : ∃! x : ℝ, (x / 5 + 4 = x / 4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l253_25350


namespace NUMINAMATH_CALUDE_drink_volume_theorem_l253_25392

/-- Represents the parts of each ingredient in the drink recipe. -/
structure DrinkRecipe where
  coke : ℕ
  sprite : ℕ
  mountainDew : ℕ
  drPepper : ℕ
  fanta : ℕ

/-- Calculates the total parts in a drink recipe. -/
def totalParts (recipe : DrinkRecipe) : ℕ :=
  recipe.coke + recipe.sprite + recipe.mountainDew + recipe.drPepper + recipe.fanta

/-- Theorem stating that given the specific drink recipe and the amount of Coke,
    the total volume of the drink is 48 ounces. -/
theorem drink_volume_theorem (recipe : DrinkRecipe)
    (h1 : recipe.coke = 4)
    (h2 : recipe.sprite = 2)
    (h3 : recipe.mountainDew = 5)
    (h4 : recipe.drPepper = 3)
    (h5 : recipe.fanta = 2)
    (h6 : 12 = recipe.coke * 3) :
    (totalParts recipe) * 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_drink_volume_theorem_l253_25392


namespace NUMINAMATH_CALUDE_max_value_of_f_l253_25388

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-3/2) 2

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ interval, f a x ≤ 1) ∧
  (∃ x ∈ interval, f a x = 1) ↔
  (a = 3/4 ∨ a = (-3-2*Real.sqrt 2)/2) := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l253_25388


namespace NUMINAMATH_CALUDE_quadratic_problem_l253_25377

def quadratic_function (b c : ℝ) : ℝ → ℝ := λ x => x^2 + b*x + c

theorem quadratic_problem (b c : ℝ) :
  (∀ x, quadratic_function b c x < 0 ↔ 1 < x ∧ x < 3) →
  (quadratic_function b c = λ x => x^2 - 4*x + 3) ∧
  (∀ m, (∀ x, quadratic_function b c x > m*x - 1) ↔ -8 < m ∧ m < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_problem_l253_25377


namespace NUMINAMATH_CALUDE_ln_increasing_on_positive_reals_l253_25363

-- Define the open interval (0, +∞)
def openPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem ln_increasing_on_positive_reals :
  StrictMonoOn Real.log openPositiveReals :=
sorry

end NUMINAMATH_CALUDE_ln_increasing_on_positive_reals_l253_25363


namespace NUMINAMATH_CALUDE_video_game_spending_ratio_l253_25315

theorem video_game_spending_ratio (initial_amount : ℚ) (video_game_cost : ℚ) (remaining : ℚ) :
  initial_amount = 100 →
  remaining = initial_amount - video_game_cost - (1/5) * (initial_amount - video_game_cost) →
  remaining = 60 →
  video_game_cost / initial_amount = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spending_ratio_l253_25315


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l253_25317

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 5^(5^n) + 6

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_1000 : units_digit (G 1000) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l253_25317


namespace NUMINAMATH_CALUDE_expression_evaluation_l253_25301

theorem expression_evaluation :
  let x : ℚ := -2
  let expr := (1 - 2 / (x + 1)) / ((x^2 - x) / (x^2 - 1))
  expr = 3/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l253_25301


namespace NUMINAMATH_CALUDE_f_max_value_l253_25341

/-- The quadratic function f(x) = -x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- Theorem: The maximum value of f(x) = -x^2 + 2x + 3 is 4 -/
theorem f_max_value : ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l253_25341


namespace NUMINAMATH_CALUDE_furniture_assembly_time_l253_25332

theorem furniture_assembly_time 
  (num_chairs : ℕ) 
  (num_tables : ℕ) 
  (time_per_piece : ℕ) 
  (h1 : num_chairs = 4) 
  (h2 : num_tables = 4) 
  (h3 : time_per_piece = 6) : 
  (num_chairs + num_tables) * time_per_piece = 48 := by
  sorry

end NUMINAMATH_CALUDE_furniture_assembly_time_l253_25332


namespace NUMINAMATH_CALUDE_black_highest_probability_l253_25359

-- Define the bag contents
def total_balls : ℕ := 8
def white_balls : ℕ := 1
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2
def black_balls : ℕ := 3

-- Define probabilities
def prob_white : ℚ := white_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_yellow : ℚ := yellow_balls / total_balls
def prob_black : ℚ := black_balls / total_balls

-- Theorem statement
theorem black_highest_probability :
  prob_black > prob_white ∧ 
  prob_black > prob_red ∧ 
  prob_black > prob_yellow :=
sorry

end NUMINAMATH_CALUDE_black_highest_probability_l253_25359


namespace NUMINAMATH_CALUDE_average_of_ABCD_l253_25338

theorem average_of_ABCD (A B C D : ℚ) 
  (eq1 : 1001 * C - 2004 * A = 4008)
  (eq2 : 1001 * B + 3005 * A - 1001 * D = 6010) :
  (A + B + C + D) / 4 = (5 + D) / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ABCD_l253_25338


namespace NUMINAMATH_CALUDE_problem_1_l253_25391

theorem problem_1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1/3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l253_25391


namespace NUMINAMATH_CALUDE_sector_central_angle_l253_25356

/-- Given a circular sector with area 1 and perimeter 4, its central angle is 2 radians -/
theorem sector_central_angle (S : ℝ) (P : ℝ) (α : ℝ) :
  S = 1 →  -- area of the sector
  P = 4 →  -- perimeter of the sector
  S = (1/2) * α * (P - α)^2 / α^2 →  -- area formula for a sector
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l253_25356


namespace NUMINAMATH_CALUDE_min_major_axis_length_l253_25302

/-- The line l: x + y - 4 = 0 -/
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

/-- The ellipse x²/16 + y²/12 = 1 -/
def ellipse_e (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- A point M on line l -/
def point_on_line_l (M : ℝ × ℝ) : Prop := line_l M.1 M.2

/-- One focus of the ellipse e -/
def focus_of_ellipse_e : ℝ × ℝ := (-2, 0)

/-- An ellipse passing through M with one focus being a focus of ellipse e -/
def new_ellipse (M : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  point_on_line_l M ∧ F = focus_of_ellipse_e

/-- The length of the major axis of an ellipse -/
noncomputable def major_axis_length (M F : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the minimum length of the major axis -/
theorem min_major_axis_length :
  ∀ M : ℝ × ℝ, new_ellipse M focus_of_ellipse_e →
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 10 ∧
  ∀ F : ℝ × ℝ, new_ellipse M F →
  major_axis_length M F ≥ min_length :=
sorry

end NUMINAMATH_CALUDE_min_major_axis_length_l253_25302


namespace NUMINAMATH_CALUDE_cos_probability_l253_25395

/-- The probability that cos(πx/2) is between 0 and 1/2 when x is randomly selected from [-1, 1] -/
theorem cos_probability : 
  ∃ (P : Set ℝ → ℝ), 
    (∀ x ∈ Set.Icc (-1) 1, P {y | 0 ≤ Real.cos (π * y / 2) ∧ Real.cos (π * y / 2) ≤ 1/2} = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_cos_probability_l253_25395


namespace NUMINAMATH_CALUDE_cos_shift_equivalence_l253_25384

theorem cos_shift_equivalence (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.cos (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equivalence_l253_25384


namespace NUMINAMATH_CALUDE_f_properties_l253_25337

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x + a - 3) / Real.log a

/-- Theorem stating the properties of f(x) -/
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (Function.Injective (f a) ↔ (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l253_25337


namespace NUMINAMATH_CALUDE_pebble_difference_l253_25371

theorem pebble_difference (candy_pebbles : ℕ) (lance_multiplier : ℕ) : 
  candy_pebbles = 4 →
  lance_multiplier = 3 →
  lance_multiplier * candy_pebbles - candy_pebbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_pebble_difference_l253_25371


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l253_25311

theorem sufficient_not_necessary (a b : ℝ) : 
  (a > b ∧ b > 0) → (1 / a < 1 / b) ∧ 
  ¬(∀ a b : ℝ, (1 / a < 1 / b) → (a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l253_25311


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l253_25323

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ a + c > b

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    a < b ∧ b < c →  -- scalene condition
    is_prime a ∧ is_prime b ∧ is_prime c →  -- prime side lengths
    a = 5 →  -- smallest side is 5
    triangle_inequality a b c →  -- valid triangle
    is_prime (a + b + c) →  -- prime perimeter
    a + b + c ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l253_25323


namespace NUMINAMATH_CALUDE_bd_squared_equals_36_l253_25304

theorem bd_squared_equals_36 
  (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_bd_squared_equals_36_l253_25304


namespace NUMINAMATH_CALUDE_train_length_l253_25321

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h1 : speed_kmph = 90) (h2 : time_sec = 5) :
  speed_kmph * (1000 / 3600) * time_sec = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l253_25321


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l253_25361

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

def a : List Bool := [true, false, true, true, false, true]  -- 101101₂
def b : List Bool := [true, true, true]  -- 111₂
def c : List Bool := [false, true, true, false, false, true, true]  -- 1100110₂
def d : List Bool := [false, true, false, true]  -- 1010₂
def result : List Bool := [true, false, true, true, true, false, true, true]  -- 11011101₂

theorem binary_addition_subtraction :
  nat_to_binary ((binary_to_nat a + binary_to_nat b + binary_to_nat c) - binary_to_nat d) = result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l253_25361


namespace NUMINAMATH_CALUDE_volume_of_given_prism_l253_25305

/-- Represents the dimensions of a rectangular prism in centimeters -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism given its dimensions -/
def prismVolume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the specific rectangular prism in the problem -/
def givenPrism : PrismDimensions :=
  { length := 4
    width := 2
    height := 8 }

/-- Theorem stating that the volume of the given rectangular prism is 64 cubic centimeters -/
theorem volume_of_given_prism :
  prismVolume givenPrism = 64 := by
  sorry

#check volume_of_given_prism

end NUMINAMATH_CALUDE_volume_of_given_prism_l253_25305


namespace NUMINAMATH_CALUDE_jeans_cost_l253_25365

theorem jeans_cost (total_cost coat_cost shoe_cost : ℕ) (h1 : total_cost = 110) (h2 : coat_cost = 40) (h3 : shoe_cost = 30) : 
  ∃ (jeans_cost : ℕ), jeans_cost * 2 + coat_cost + shoe_cost = total_cost ∧ jeans_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_l253_25365


namespace NUMINAMATH_CALUDE_sin_300_degrees_l253_25328

theorem sin_300_degrees : Real.sin (300 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l253_25328


namespace NUMINAMATH_CALUDE_xiao_ming_tasks_minimum_time_l253_25340

def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

def minimum_time : ℕ := 85

theorem xiao_ming_tasks_minimum_time :
  minimum_time = max review_time (max rest_time homework_time) :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_tasks_minimum_time_l253_25340


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l253_25330

theorem decimal_to_fraction : (0.38 : ℚ) = 19 / 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l253_25330


namespace NUMINAMATH_CALUDE_intersection_point_sum_l253_25385

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 = 4*y
def C2 (x y : ℝ) : Prop := x + y = 5

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem intersection_point_sum : 
  1 / distance P A + 1 / distance P B = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l253_25385


namespace NUMINAMATH_CALUDE_systematic_sampling_correct_l253_25326

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  populationSize : ℕ
  sampleSize : ℕ
  firstItem : ℕ
  samplingInterval : ℕ

/-- Generates the sample based on the systematic sampling scheme -/
def generateSample (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.firstItem + i * s.samplingInterval)

/-- Theorem: The systematic sampling for the given problem yields the correct sample -/
theorem systematic_sampling_correct :
  let s : SystematicSampling := {
    populationSize := 50,
    sampleSize := 5,
    firstItem := 7,
    samplingInterval := 10
  }
  generateSample s = [7, 17, 27, 37, 47] := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_correct_l253_25326


namespace NUMINAMATH_CALUDE_doris_eggs_l253_25396

/-- Represents the number of eggs in a package -/
inductive EggPackage
  | small : EggPackage
  | large : EggPackage

/-- Returns the number of eggs in a package -/
def eggs_in_package (p : EggPackage) : Nat :=
  match p with
  | EggPackage.small => 6
  | EggPackage.large => 11

/-- Calculates the total number of eggs bought given the number of large packs -/
def total_eggs (large_packs : Nat) : Nat :=
  large_packs * eggs_in_package EggPackage.large

/-- Proves that Doris bought 55 eggs in total -/
theorem doris_eggs :
  total_eggs 5 = 55 := by sorry

end NUMINAMATH_CALUDE_doris_eggs_l253_25396


namespace NUMINAMATH_CALUDE_max_value_theorem_l253_25352

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2*x*y + 3*y^2 = 12) :
  ∃ (M : ℝ), M = 24 + 24*Real.sqrt 3 ∧ x^2 + 2*x*y + 3*y^2 ≤ M ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 - 2*x'*y' + 3*y'^2 = 12 ∧ x'^2 + 2*x'*y' + 3*y'^2 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l253_25352


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l253_25300

/-- The trajectory of the midpoint of a line segment PQ, where P is fixed at (4, 0) and Q is on the circle x^2 + y^2 = 4 -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (qx qy : ℝ), qx^2 + qy^2 = 4 ∧ x = (4 + qx) / 2 ∧ y = qy / 2) → 
  (x - 2)^2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l253_25300


namespace NUMINAMATH_CALUDE_line_segment_length_l253_25324

/-- The length of a line segment with endpoints (1,4) and (8,16) is √193. -/
theorem line_segment_length : Real.sqrt 193 = Real.sqrt ((8 - 1)^2 + (16 - 4)^2) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l253_25324


namespace NUMINAMATH_CALUDE_salary_increase_proof_l253_25397

theorem salary_increase_proof (original_salary : ℝ) 
  (h1 : original_salary * 1.8 = 25000) 
  (h2 : original_salary > 0) : 
  25000 - original_salary = 11111.11 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l253_25397


namespace NUMINAMATH_CALUDE_total_initials_eq_thousand_l253_25387

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def initials_length : ℕ := 3

/-- The total number of possible three-letter sets of initials using letters A through J -/
def total_initials : ℕ := num_letters ^ initials_length

/-- Theorem stating that the total number of possible three-letter sets of initials
    using letters A through J is 1000 -/
theorem total_initials_eq_thousand : total_initials = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_initials_eq_thousand_l253_25387


namespace NUMINAMATH_CALUDE_cafeteria_fruit_sale_l253_25380

/-- Cafeteria fruit sale problem -/
theorem cafeteria_fruit_sale
  (initial_apples : ℕ) 
  (initial_oranges : ℕ) 
  (apple_price : ℚ) 
  (orange_price : ℚ) 
  (total_earnings : ℚ) 
  (apples_left : ℕ) 
  (h1 : initial_apples = 50)
  (h2 : initial_oranges = 40)
  (h3 : apple_price = 4/5)
  (h4 : orange_price = 1/2)
  (h5 : total_earnings = 49)
  (h6 : apples_left = 10) :
  initial_oranges - (total_earnings - (initial_apples - apples_left) * apple_price) / orange_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_fruit_sale_l253_25380
