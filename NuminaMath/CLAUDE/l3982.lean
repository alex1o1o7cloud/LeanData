import Mathlib

namespace nine_fifteen_div_fifty_four_five_l3982_398220

theorem nine_fifteen_div_fifty_four_five :
  (9 : ℝ)^15 / 54^5 = 1594323 * (3 : ℝ)^(1/3) := by sorry

end nine_fifteen_div_fifty_four_five_l3982_398220


namespace journey_distance_l3982_398228

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = 224 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 :=
by
  sorry

end journey_distance_l3982_398228


namespace blue_face_area_l3982_398241

-- Define a tetrahedron with right-angled edges
structure RightAngledTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  red_area : ℝ
  yellow_area : ℝ
  green_area : ℝ
  blue_area : ℝ
  right_angle_condition : a^2 + b^2 = c^2
  red_area_condition : red_area = (1/2) * a * b
  yellow_area_condition : yellow_area = (1/2) * b * c
  green_area_condition : green_area = (1/2) * c * a
  blue_area_condition : blue_area = (1/2) * (a^2 + b^2 + c^2)

-- Theorem statement
theorem blue_face_area (t : RightAngledTetrahedron) 
  (h1 : t.red_area = 60) 
  (h2 : t.yellow_area = 20) 
  (h3 : t.green_area = 15) : 
  t.blue_area = 65 := by
  sorry


end blue_face_area_l3982_398241


namespace rectangular_to_cylindrical_l3982_398256

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 6 ∧
  θ = 5 * π / 3 ∧
  z = 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ := by
sorry

end rectangular_to_cylindrical_l3982_398256


namespace circle_equation_theta_range_l3982_398200

theorem circle_equation_theta_range :
  ∀ (x y θ : ℝ),
  (x^2 + y^2 + x + Real.sqrt 3 * y + Real.tan θ = 0) →
  (-π/2 < θ ∧ θ < π/2) →
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ↔ 
    p.1^2 + p.2^2 + p.1 + Real.sqrt 3 * p.2 + Real.tan θ = 0) →
  -π/2 < θ ∧ θ < π/4 :=
by sorry

end circle_equation_theta_range_l3982_398200


namespace octagon_arc_length_l3982_398201

/-- The arc length intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (side_length : ℝ) (h : side_length = 4) :
  let radius : ℝ := side_length
  let circumference : ℝ := 2 * π * radius
  let central_angle : ℝ := π / 4  -- 45 degrees in radians
  let arc_length : ℝ := (central_angle / (2 * π)) * circumference
  arc_length = π :=
by sorry

end octagon_arc_length_l3982_398201


namespace square_sum_of_reciprocal_sum_and_sum_l3982_398205

theorem square_sum_of_reciprocal_sum_and_sum (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) (h2 : x + y = 5) : x^2 + y^2 = 35/2 := by
  sorry

end square_sum_of_reciprocal_sum_and_sum_l3982_398205


namespace degree_to_radian_conversion_l3982_398223

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  (-300 : ℝ) * π / 180 = -5 * π / 3 :=
by sorry

end degree_to_radian_conversion_l3982_398223


namespace butterflies_in_garden_l3982_398243

theorem butterflies_in_garden (initial : ℕ) (remaining : ℕ) : 
  remaining = 6 ∧ 3 * remaining = 2 * initial → initial = 9 := by
  sorry

end butterflies_in_garden_l3982_398243


namespace jerry_syrup_time_l3982_398277

/-- Represents the time it takes Jerry to make cherry syrup -/
def make_cherry_syrup (cherries_per_quart : ℕ) (picking_time : ℕ) (picking_amount : ℕ) (syrup_making_time : ℕ) (quarts : ℕ) : ℕ :=
  let picking_rate : ℚ := picking_amount / picking_time
  let total_cherries : ℕ := cherries_per_quart * quarts
  let total_picking_time : ℕ := (total_cherries / picking_rate).ceil.toNat
  total_picking_time + syrup_making_time

/-- Proves that it takes Jerry 33 hours to make 9 quarts of cherry syrup -/
theorem jerry_syrup_time :
  make_cherry_syrup 500 2 300 3 9 = 33 := by
  sorry

end jerry_syrup_time_l3982_398277


namespace odd_number_grouping_l3982_398297

theorem odd_number_grouping (n : ℕ) (odd_number : ℕ) : 
  (odd_number = 2007) →
  (∀ k : ℕ, k < n → (k^2 < 1004 ∧ 1004 ≤ (k+1)^2)) →
  (n = 32) :=
by sorry

end odd_number_grouping_l3982_398297


namespace total_dress_cost_l3982_398237

theorem total_dress_cost (pauline_dress : ℕ) (h1 : pauline_dress = 30)
  (jean_dress : ℕ) (h2 : jean_dress = pauline_dress - 10)
  (ida_dress : ℕ) (h3 : ida_dress = jean_dress + 30)
  (patty_dress : ℕ) (h4 : patty_dress = ida_dress + 10) :
  pauline_dress + jean_dress + ida_dress + patty_dress = 160 := by
sorry

end total_dress_cost_l3982_398237


namespace gcd_problems_l3982_398295

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 440 556 = 4) := by
  sorry

end gcd_problems_l3982_398295


namespace jake_weight_loss_l3982_398231

theorem jake_weight_loss (jake_current : ℕ) (combined_weight : ℕ) (weight_loss : ℕ) : 
  jake_current = 198 →
  combined_weight = 293 →
  jake_current - weight_loss = 2 * (combined_weight - jake_current) →
  weight_loss = 8 := by
  sorry

end jake_weight_loss_l3982_398231


namespace stating_table_tennis_sequences_count_l3982_398215

/-- Represents the number of possible game sequences in a table tennis match --/
def table_tennis_sequences : ℕ := 20

/-- 
Theorem stating that the number of possible game sequences in a table tennis match,
where the first player to win three games wins the match, is exactly 20.
--/
theorem table_tennis_sequences_count : table_tennis_sequences = 20 := by
  sorry

end stating_table_tennis_sequences_count_l3982_398215


namespace wily_person_exists_l3982_398246

inductive PersonType
  | Knight
  | Liar
  | Wily

structure Person where
  type : PersonType
  statement : Prop

def is_truthful (p : Person) : Prop :=
  match p.type with
  | PersonType.Knight => p.statement
  | PersonType.Liar => ¬p.statement
  | PersonType.Wily => True

theorem wily_person_exists (people : Fin 3 → Person)
  (h1 : (people 0).statement = ∃ i, (people i).type = PersonType.Liar)
  (h2 : (people 1).statement = ∀ i j, i ≠ j → ((people i).type = PersonType.Liar ∨ (people j).type = PersonType.Liar))
  (h3 : (people 2).statement = ∀ i, (people i).type = PersonType.Liar)
  : ∃ i, (people i).type = PersonType.Wily :=
by
  sorry

end wily_person_exists_l3982_398246


namespace bert_sandwiches_remaining_l3982_398285

def sandwiches_remaining (initial : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  initial - (first_day + second_day)

theorem bert_sandwiches_remaining :
  let initial := 12
  let first_day := initial / 2
  let second_day := first_day - 2
  sandwiches_remaining initial first_day second_day = 2 := by
sorry

end bert_sandwiches_remaining_l3982_398285


namespace running_yardage_l3982_398260

/-- The star running back's total yardage -/
def total_yardage : ℕ := 150

/-- The star running back's passing yardage -/
def passing_yardage : ℕ := 60

/-- Theorem: The star running back's running yardage is 90 yards -/
theorem running_yardage : total_yardage - passing_yardage = 90 := by
  sorry

end running_yardage_l3982_398260


namespace arithmetic_sequence_2_to_2014_l3982_398263

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting with 2, ending with 2014, 
    and having a common difference of 4 contains exactly 504 terms -/
theorem arithmetic_sequence_2_to_2014 : 
  arithmetic_sequence_length 2 2014 4 = 504 := by
  sorry

end arithmetic_sequence_2_to_2014_l3982_398263


namespace jony_start_time_l3982_398290

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculate the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Represents Jony's walk -/
structure Walk where
  startBlock : Nat
  turnaroundBlock : Nat
  endBlock : Nat
  blockLength : Nat
  speed : Nat
  endTime : Time

theorem jony_start_time (w : Walk) (h1 : w.startBlock = 10)
    (h2 : w.turnaroundBlock = 90) (h3 : w.endBlock = 70)
    (h4 : w.blockLength = 40) (h5 : w.speed = 100)
    (h6 : w.endTime = ⟨7, 40⟩) :
    timeDifference w.endTime ⟨7, 0⟩ =
      ((w.turnaroundBlock - w.startBlock + w.turnaroundBlock - w.endBlock) * w.blockLength) / w.speed :=
  sorry

end jony_start_time_l3982_398290


namespace consecutive_hits_arrangements_eq_30_l3982_398225

/-- Represents the number of ways to arrange 4 hits in 8 shots with exactly two consecutive hits -/
def consecutive_hits_arrangements : ℕ :=
  let total_shots : ℕ := 8
  let hits : ℕ := 4
  let misses : ℕ := total_shots - hits
  let spaces : ℕ := misses + 1
  let ways_to_place_consecutive_pair : ℕ := spaces
  let remaining_spaces : ℕ := spaces - 1
  let remaining_hits : ℕ := hits - 2
  let ways_to_place_remaining_hits : ℕ := Nat.choose remaining_spaces remaining_hits
  ways_to_place_consecutive_pair * ways_to_place_remaining_hits

/-- Theorem stating that the number of arrangements is 30 -/
theorem consecutive_hits_arrangements_eq_30 : consecutive_hits_arrangements = 30 := by
  sorry

end consecutive_hits_arrangements_eq_30_l3982_398225


namespace project_hours_calculation_l3982_398249

theorem project_hours_calculation (kate : ℕ) (pat : ℕ) (mark : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : 3 * pat = mark)
  (h3 : mark = kate + 85) :
  kate + pat + mark = 153 := by
  sorry

end project_hours_calculation_l3982_398249


namespace max_intersections_three_circles_one_line_l3982_398238

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The total maximum number of intersection points -/
def total_max_intersections : ℕ := max_circle_intersections + max_line_circle_intersections

theorem max_intersections_three_circles_one_line :
  total_max_intersections = 12 := by sorry

end max_intersections_three_circles_one_line_l3982_398238


namespace smallest_regiment_size_exact_smallest_regiment_size_new_uniforms_condition_l3982_398209

theorem smallest_regiment_size (m n : ℕ) (h1 : m ≥ 40) (h2 : n ≥ 30) : m * n ≥ 1200 := by
  sorry

theorem exact_smallest_regiment_size : ∃ m n : ℕ, m ≥ 40 ∧ n ≥ 30 ∧ m * n = 1200 := by
  sorry

theorem new_uniforms_condition (m n : ℕ) (h1 : m ≥ 40) (h2 : n ≥ 30) :
  (m * n : ℚ) / 100 ≥ (0.3 : ℚ) * m ∧ (m * n : ℚ) / 100 ≥ (0.4 : ℚ) * n := by
  sorry

end smallest_regiment_size_exact_smallest_regiment_size_new_uniforms_condition_l3982_398209


namespace stream_speed_l3982_398235

/-- Proves that the speed of a stream is 5 km/h, given a man's swimming speed in still water
    and the relative time taken to swim upstream vs downstream. -/
theorem stream_speed (man_speed : ℝ) (upstream_time_ratio : ℝ) 
  (h1 : man_speed = 15)
  (h2 : upstream_time_ratio = 2) : 
  ∃ (stream_speed : ℝ), stream_speed = 5 ∧
  (man_speed + stream_speed) * 1 = (man_speed - stream_speed) * upstream_time_ratio :=
by sorry

end stream_speed_l3982_398235


namespace dress_price_ratio_l3982_398266

theorem dress_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_ratio : ℝ := 2/3
  let cost : ℝ := cost_ratio * selling_price
  cost / marked_price = 1/2 := by
sorry

end dress_price_ratio_l3982_398266


namespace two_truth_tellers_l3982_398289

/-- Represents the four Knaves -/
inductive Knave : Type
  | Hearts
  | Clubs
  | Diamonds
  | Spades

/-- Represents whether a Knave is telling the truth or lying -/
def Truthfulness : Type := Knave → Bool

/-- A consistent arrangement of truthfulness satisfies the interdependence of Knaves' statements -/
def is_consistent (t : Truthfulness) : Prop :=
  t Knave.Hearts = (t Knave.Clubs = false ∧ t Knave.Diamonds = true ∧ t Knave.Spades = false)

/-- Counts the number of truth-telling Knaves -/
def count_truth_tellers (t : Truthfulness) : Nat :=
  (if t Knave.Hearts then 1 else 0) +
  (if t Knave.Clubs then 1 else 0) +
  (if t Knave.Diamonds then 1 else 0) +
  (if t Knave.Spades then 1 else 0)

/-- Main theorem: Any consistent arrangement has exactly two truth-tellers -/
theorem two_truth_tellers (t : Truthfulness) (h : is_consistent t) :
  count_truth_tellers t = 2 := by
  sorry

end two_truth_tellers_l3982_398289


namespace even_function_iff_b_zero_l3982_398276

/-- For real numbers a and b, and function f(x) = a*cos(x) + b*sin(x),
    f(x) is an even function if and only if b = 0 -/
theorem even_function_iff_b_zero (a b : ℝ) :
  (∀ x, a * Real.cos x + b * Real.sin x = a * Real.cos (-x) + b * Real.sin (-x)) ↔ b = 0 :=
by sorry

end even_function_iff_b_zero_l3982_398276


namespace fred_cantaloupes_l3982_398298

theorem fred_cantaloupes (keith_cantaloupes jason_cantaloupes total_cantaloupes : ℕ)
  (h1 : keith_cantaloupes = 29)
  (h2 : jason_cantaloupes = 20)
  (h3 : total_cantaloupes = 65)
  (h4 : ∃ fred_cantaloupes : ℕ, keith_cantaloupes + jason_cantaloupes + fred_cantaloupes = total_cantaloupes) :
  ∃ fred_cantaloupes : ℕ, fred_cantaloupes = 16 :=
by
  sorry

end fred_cantaloupes_l3982_398298


namespace sector_radius_l3982_398224

/-- Given a circular sector with area 13.75 cm² and arc length 5.5 cm, the radius is 5 cm -/
theorem sector_radius (area : Real) (arc_length : Real) (radius : Real) :
  area = 13.75 ∧ arc_length = 5.5 ∧ area = (1/2) * radius * arc_length →
  radius = 5 := by
sorry

end sector_radius_l3982_398224


namespace sum_of_fractions_less_than_target_l3982_398292

theorem sum_of_fractions_less_than_target : 
  (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-9/20 : ℚ) < (45/100 : ℚ) :=
by sorry

end sum_of_fractions_less_than_target_l3982_398292


namespace smallest_two_digit_prime_with_reversed_composite_l3982_398222

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem smallest_two_digit_prime_with_reversed_composite :
  ∃ n : ℕ,
    is_two_digit n ∧
    is_prime n ∧
    (n / 10 ≥ 3) ∧
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → is_prime m → (m / 10 ≥ 3) → is_composite (reverse_digits m) → n ≤ m) ∧
    n = 41 :=
  sorry

end smallest_two_digit_prime_with_reversed_composite_l3982_398222


namespace equation_identity_l3982_398229

theorem equation_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end equation_identity_l3982_398229


namespace tan_sin_expression_simplification_l3982_398244

theorem tan_sin_expression_simplification :
  Real.tan (70 * π / 180) * Real.sin (80 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_sin_expression_simplification_l3982_398244


namespace equilateral_triangles_in_decagon_l3982_398226

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Count of distinct equilateral triangles with at least two vertices in a regular polygon -/
def count_equilateral_triangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem equilateral_triangles_in_decagon :
  ∃ (decagon : RegularPolygon 10),
    count_equilateral_triangles 10 decagon = 82 :=
  sorry

end equilateral_triangles_in_decagon_l3982_398226


namespace total_pages_read_l3982_398269

theorem total_pages_read (pages_yesterday pages_today : ℕ) 
  (h1 : pages_yesterday = 21) 
  (h2 : pages_today = 17) : 
  pages_yesterday + pages_today = 38 := by
  sorry

end total_pages_read_l3982_398269


namespace walter_bus_time_l3982_398236

def wake_up_time : Nat := 6 * 60
def leave_time : Nat := 7 * 60
def return_time : Nat := 16 * 60 + 30
def num_classes : Nat := 7
def class_duration : Nat := 45
def lunch_duration : Nat := 45
def additional_time : Nat := 90

def total_away_time : Nat := return_time - leave_time
def total_school_time : Nat := num_classes * class_duration + lunch_duration + additional_time

theorem walter_bus_time :
  total_away_time - total_school_time = 120 := by
  sorry

end walter_bus_time_l3982_398236


namespace slips_with_three_count_l3982_398288

/-- Given a bag of slips with either 3 or 8 written on them, 
    this function calculates the expected value of a randomly drawn slip. -/
def expected_value (total_slips : ℕ) (slips_with_three : ℕ) : ℚ :=
  (3 * slips_with_three + 8 * (total_slips - slips_with_three)) / total_slips

/-- Theorem stating that given the conditions of the problem, 
    the number of slips with 3 written on them is 8. -/
theorem slips_with_three_count : 
  ∃ (x : ℕ), x ≤ 15 ∧ expected_value 15 x = 5.4 ∧ x = 8 := by
  sorry


end slips_with_three_count_l3982_398288


namespace pen_cost_is_six_l3982_398253

/-- The cost of the pen Joshua wants to buy -/
def pen_cost : ℚ := 6

/-- The amount of money Joshua has in his pocket -/
def pocket_money : ℚ := 5

/-- The amount of money Joshua borrowed from his neighbor -/
def borrowed_money : ℚ := 68 / 100

/-- The additional amount Joshua needs to buy the pen -/
def additional_money_needed : ℚ := 32 / 100

/-- Theorem stating that the cost of the pen is $6.00 -/
theorem pen_cost_is_six :
  pen_cost = pocket_money + borrowed_money + additional_money_needed :=
by sorry

end pen_cost_is_six_l3982_398253


namespace coords_wrt_origin_invariant_point_P_coords_l3982_398273

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Coordinates of a point with respect to the origin -/
def coordsWrtOrigin (p : Point) : ℝ × ℝ := (p.x, p.y)

theorem coords_wrt_origin_invariant (p : Point) :
  coordsWrtOrigin p = (p.x, p.y) := by sorry

theorem point_P_coords :
  let P : Point := ⟨-1, -3⟩
  coordsWrtOrigin P = (-1, -3) := by sorry

end coords_wrt_origin_invariant_point_P_coords_l3982_398273


namespace perp_plane_necessary_not_sufficient_l3982_398271

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the "line in plane" relation
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem perp_plane_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧
  ¬(perp_planes α β → perp_line_plane m β) :=
sorry

end perp_plane_necessary_not_sufficient_l3982_398271


namespace cos_equality_implies_70_l3982_398240

theorem cos_equality_implies_70 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) 
  (h3 : Real.cos (n * π / 180) = Real.cos (1010 * π / 180)) : n = 70 := by
  sorry

end cos_equality_implies_70_l3982_398240


namespace min_white_pairs_problem_solution_l3982_398216

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Nat)

/-- Calculates the total number of adjacent cell pairs in a square grid -/
def total_pairs (g : Grid) : Nat :=
  2 * g.size * (g.size - 1)

/-- Calculates the maximum number of central cell pairs -/
def max_central_pairs (g : Grid) : Nat :=
  ((g.size - 2) * (g.size - 2)) / 2

/-- Theorem: Given an 8x8 grid with 20 black cells, the minimum number of pairs of adjacent white cells is 34 -/
theorem min_white_pairs (g : Grid) (h1 : g.size = 8) (h2 : g.black_cells = 20) :
  total_pairs g - (60 + min g.black_cells (max_central_pairs g)) = 34 := by
  sorry

/-- Main theorem stating the result for the specific problem -/
theorem problem_solution : 
  ∃ (g : Grid), g.size = 8 ∧ g.black_cells = 20 ∧ 
  (total_pairs g - (60 + min g.black_cells (max_central_pairs g)) = 34) := by
  sorry

end min_white_pairs_problem_solution_l3982_398216


namespace mixed_doubles_handshakes_l3982_398291

/-- Represents a mixed doubles tennis tournament -/
structure MixedDoublesTournament where
  teams : Nat
  players_per_team : Nat
  opposite_gender_players : Nat

/-- Calculates the number of handshakes in a mixed doubles tournament -/
def handshakes (tournament : MixedDoublesTournament) : Nat :=
  tournament.teams * (tournament.opposite_gender_players - 1)

/-- Theorem: In a mixed doubles tennis tournament with 4 teams, 
    where each player shakes hands once with every player of the 
    opposite gender except their own partner, the total number 
    of handshakes is 12. -/
theorem mixed_doubles_handshakes :
  let tournament : MixedDoublesTournament := {
    teams := 4,
    players_per_team := 2,
    opposite_gender_players := 4
  }
  handshakes tournament = 12 := by
  sorry

end mixed_doubles_handshakes_l3982_398291


namespace average_weight_decrease_l3982_398252

/-- Given a group of people and a new person joining, calculate the decrease in average weight -/
theorem average_weight_decrease (initial_count : ℕ) (initial_average : ℝ) (new_person_weight : ℝ) : 
  initial_count = 20 →
  initial_average = 55 →
  new_person_weight = 50 →
  let total_weight := initial_count * initial_average
  let new_total_weight := total_weight + new_person_weight
  let new_count := initial_count + 1
  let new_average := new_total_weight / new_count
  abs (initial_average - new_average - 0.24) < 0.01 := by
sorry

end average_weight_decrease_l3982_398252


namespace teacher_discount_l3982_398258

theorem teacher_discount (students : ℕ) (pens_per_student : ℕ) (notebooks_per_student : ℕ) 
  (binders_per_student : ℕ) (highlighters_per_student : ℕ) 
  (pen_cost : ℚ) (notebook_cost : ℚ) (binder_cost : ℚ) (highlighter_cost : ℚ) 
  (amount_spent : ℚ) : 
  students = 30 →
  pens_per_student = 5 →
  notebooks_per_student = 3 →
  binders_per_student = 1 →
  highlighters_per_student = 2 →
  pen_cost = 1/2 →
  notebook_cost = 5/4 →
  binder_cost = 17/4 →
  highlighter_cost = 3/4 →
  amount_spent = 260 →
  (students * pens_per_student * pen_cost + 
   students * notebooks_per_student * notebook_cost + 
   students * binders_per_student * binder_cost + 
   students * highlighters_per_student * highlighter_cost) - amount_spent = 100 := by
  sorry

end teacher_discount_l3982_398258


namespace last_term_is_123_l3982_398279

/-- A sequence of natural numbers -/
def Sequence : Type := ℕ → ℕ

/-- The specific sequence from the problem -/
def s : Sequence :=
  fun n =>
    match n with
    | 1 => 2
    | 2 => 3
    | 3 => 6
    | 4 => 15
    | 5 => 33
    | 6 => 123
    | _ => 0  -- For completeness, though we only care about the first 6 terms

/-- The theorem stating that the last (6th) term of the sequence is 123 -/
theorem last_term_is_123 : s 6 = 123 := by
  sorry


end last_term_is_123_l3982_398279


namespace square_root_of_product_l3982_398245

theorem square_root_of_product : Real.sqrt ((90 + 6) * (90 - 6)) = 90 := by
  sorry

end square_root_of_product_l3982_398245


namespace quadratic_roots_condition_l3982_398272

theorem quadratic_roots_condition (r : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (r - 4) * x₁^2 - 2*(r - 3) * x₁ + r = 0 ∧
   (r - 4) * x₂^2 - 2*(r - 3) * x₂ + r = 0 ∧
   x₁ > -1 ∧ x₂ > -1) ↔ 
  (3.5 < r ∧ r < 4.5) :=
by sorry

end quadratic_roots_condition_l3982_398272


namespace min_value_constrained_l3982_398265

theorem min_value_constrained (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ m : ℝ, ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ m) ∧
  (∀ ε > 0, ∃ x y : ℝ, a * x^2 + b * y^2 = 1 ∧ c * x + d * y^2 < -c / Real.sqrt a + ε) :=
sorry

end min_value_constrained_l3982_398265


namespace squirrels_and_nuts_l3982_398206

theorem squirrels_and_nuts (squirrels : ℕ) (nuts : ℕ) : 
  squirrels = 4 → squirrels - nuts = 2 → nuts = 2 := by
  sorry

end squirrels_and_nuts_l3982_398206


namespace model4_best_fitting_l3982_398259

-- Define the structure for a regression model
structure RegressionModel where
  name : String
  r_squared : Float

-- Define the principle of better fitting
def better_fitting (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

-- Define the four models
def model1 : RegressionModel := ⟨"Model 1", 0.55⟩
def model2 : RegressionModel := ⟨"Model 2", 0.65⟩
def model3 : RegressionModel := ⟨"Model 3", 0.79⟩
def model4 : RegressionModel := ⟨"Model 4", 0.95⟩

-- Define a list of all models
def all_models : List RegressionModel := [model1, model2, model3, model4]

-- Theorem: Model 4 has the best fitting effect
theorem model4_best_fitting :
  ∀ m ∈ all_models, m ≠ model4 → better_fitting model4 m :=
by sorry

end model4_best_fitting_l3982_398259


namespace same_terminal_side_angle_l3982_398230

theorem same_terminal_side_angle : ∃ (θ : Real), 
  0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  ∃ (k : ℤ), θ = 2 * k * Real.pi + (-4 * Real.pi / 3) ∧
  θ = 2 * Real.pi / 3 :=
sorry

end same_terminal_side_angle_l3982_398230


namespace truck_speed_truck_speed_proof_l3982_398287

/-- Proves that a truck traveling 600 meters in 60 seconds has a speed of 36 kilometers per hour -/
theorem truck_speed : ℝ → Prop :=
  fun (speed : ℝ) =>
    let distance : ℝ := 600  -- meters
    let time : ℝ := 60       -- seconds
    let meters_per_km : ℝ := 1000
    let seconds_per_hour : ℝ := 3600
    speed = (distance / time) * (seconds_per_hour / meters_per_km) → speed = 36

/-- The actual speed of the truck in km/h -/
def actual_speed : ℝ := 36

theorem truck_speed_proof : truck_speed actual_speed :=
  sorry

end truck_speed_truck_speed_proof_l3982_398287


namespace alices_preferred_number_l3982_398234

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem alices_preferred_number (n : ℕ) 
  (h1 : 70 < n ∧ n < 140)
  (h2 : n % 13 = 0)
  (h3 : n % 3 ≠ 0)
  (h4 : sum_of_digits n % 4 = 0) :
  n = 130 := by
sorry

end alices_preferred_number_l3982_398234


namespace exists_set_with_150_primes_l3982_398239

/-- The number of primes less than 1000 -/
def primes_lt_1000 : ℕ := 168

/-- Function that counts the number of primes in a set of 2002 consecutive integers starting from n -/
def count_primes (n : ℕ) : ℕ := sorry

theorem exists_set_with_150_primes :
  ∃ n : ℕ, count_primes n = 150 :=
sorry

end exists_set_with_150_primes_l3982_398239


namespace parametric_eq_line_l3982_398211

/-- Prove that the parametric equations x = t - 1 and y = 2t - 1 represent the line y = 2x + 1 for all real values of t. -/
theorem parametric_eq_line (t : ℝ) : 
  let x := t - 1
  let y := 2*t - 1
  y = 2*x + 1 := by
  sorry

end parametric_eq_line_l3982_398211


namespace magic_square_solution_l3982_398219

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℚ)
  (row_sum : ℚ)
  (magic_property : 
    a11 + a12 + a13 = row_sum ∧
    a21 + a22 + a23 = row_sum ∧
    a31 + a32 + a33 = row_sum ∧
    a11 + a21 + a31 = row_sum ∧
    a12 + a22 + a32 = row_sum ∧
    a13 + a23 + a33 = row_sum ∧
    a11 + a22 + a33 = row_sum ∧
    a13 + a22 + a31 = row_sum)

/-- The theorem stating the solution to the magic square problem -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.a12 = 25)
  (h2 : ms.a13 = 64)
  (h3 : ms.a21 = 3) :
  ms.a11 = 272 / 3 := by
  sorry

end magic_square_solution_l3982_398219


namespace complex_roots_theorem_l3982_398250

theorem complex_roots_theorem (p q r : ℂ) 
  (sum_eq : p + q + r = -1)
  (sum_prod_eq : p*q + p*r + q*r = -1)
  (prod_eq : p*q*r = -1) :
  (p = -1 ∧ q = 1 ∧ r = 1) ∨
  (p = -1 ∧ q = 1 ∧ r = 1) ∨
  (p = 1 ∧ q = -1 ∧ r = 1) ∨
  (p = 1 ∧ q = 1 ∧ r = -1) ∨
  (p = 1 ∧ q = -1 ∧ r = 1) ∨
  (p = -1 ∧ q = 1 ∧ r = 1) := by
  sorry

end complex_roots_theorem_l3982_398250


namespace alice_favorite_number_l3982_398274

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  70 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  ¬(n % 3 = 0) ∧
  is_prime (digit_sum n)

theorem alice_favorite_number :
  ∀ n : ℕ, satisfies_conditions n ↔ n = 104 :=
sorry

end alice_favorite_number_l3982_398274


namespace least_difference_l3982_398217

theorem least_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 5 ∧
  Even x ∧
  Nat.Prime y.toNat ∧ Odd y ∧
  Odd z ∧ z % 3 = 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∀ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧
    y' - x' > 5 ∧
    Even x' ∧
    Nat.Prime y'.toNat ∧ Odd y' ∧
    Odd z' ∧ z' % 3 = 0 ∧
    x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' →
    z - x ≤ z' - x' ∧
    z - x = 13 :=
by sorry

end least_difference_l3982_398217


namespace quadratic_inequality_solution_set_l3982_398214

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 ≤ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

end quadratic_inequality_solution_set_l3982_398214


namespace largest_divisor_power_l3982_398264

-- Define pow function
def pow (n : ℕ) : ℕ :=
  sorry

-- Define X
def X : ℕ := 2310

-- Define the product of pow(n) from 2 to 5400
def product : ℕ :=
  sorry

-- Theorem statement
theorem largest_divisor_power : 
  (∃ m : ℕ, X^m ∣ product ∧ ∀ k > m, ¬(X^k ∣ product)) → 
  (∃ m : ℕ, X^m ∣ product ∧ ∀ k > m, ¬(X^k ∣ product) ∧ m = 534) :=
by sorry

end largest_divisor_power_l3982_398264


namespace gcd_of_319_377_116_l3982_398212

theorem gcd_of_319_377_116 : Nat.gcd 319 (Nat.gcd 377 116) = 29 := by
  sorry

end gcd_of_319_377_116_l3982_398212


namespace smallest_m_pair_l3982_398232

/-- Given the equation 19m + 90 + 8n = 1998, where m and n are positive integers,
    the pair (m, n) with the smallest possible value for m is (4, 229). -/
theorem smallest_m_pair : 
  ∃ (m n : ℕ), 
    (∀ (m' n' : ℕ), 19 * m' + 90 + 8 * n' = 1998 → m ≤ m') ∧ 
    19 * m + 90 + 8 * n = 1998 ∧ 
    m = 4 ∧ 
    n = 229 := by
  sorry

end smallest_m_pair_l3982_398232


namespace at_most_one_obtuse_l3982_398270

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := 90 < angle

-- Theorem statement
theorem at_most_one_obtuse (t : Triangle) : 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle2) ∧ 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle3) ∧ 
  ¬(is_obtuse t.angle2 ∧ is_obtuse t.angle3) :=
sorry

end at_most_one_obtuse_l3982_398270


namespace students_in_same_group_l3982_398208

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The probability that both students are in the same group -/
def prob_same_group : ℚ := num_groups * (prob_join_group * prob_join_group)

theorem students_in_same_group :
  prob_same_group = 1 / 3 :=
sorry

end students_in_same_group_l3982_398208


namespace student_fail_marks_l3982_398247

theorem student_fail_marks (total_marks passing_percentage student_marks : ℕ) 
  (h1 : total_marks = 700)
  (h2 : passing_percentage = 33)
  (h3 : student_marks = 175) :
  (total_marks * passing_percentage / 100 : ℕ) - student_marks = 56 :=
by sorry

end student_fail_marks_l3982_398247


namespace division_multiplication_problem_l3982_398275

theorem division_multiplication_problem : (150 : ℚ) / ((30 : ℚ) / 3) * 2 = 30 := by
  sorry

end division_multiplication_problem_l3982_398275


namespace p_plus_q_equals_twenty_one_halves_l3982_398242

theorem p_plus_q_equals_twenty_one_halves 
  (p q : ℝ) 
  (hp : p^3 - 21*p^2 + 35*p - 105 = 0) 
  (hq : 5*q^3 - 35*q^2 - 175*q + 1225 = 0) : 
  p + q = 21/2 := by sorry

end p_plus_q_equals_twenty_one_halves_l3982_398242


namespace buffet_dishes_l3982_398282

theorem buffet_dishes (mango_salsa_dishes : ℕ) (mango_jelly_dishes : ℕ) (oliver_edible_dishes : ℕ) 
  (fresh_mango_ratio : ℚ) (oliver_pick_out_dishes : ℕ) :
  mango_salsa_dishes = 3 →
  mango_jelly_dishes = 1 →
  fresh_mango_ratio = 1 / 6 →
  oliver_pick_out_dishes = 2 →
  oliver_edible_dishes = 28 →
  ∃ (total_dishes : ℕ), 
    total_dishes = 36 ∧ 
    (fresh_mango_ratio * total_dishes : ℚ).num = oliver_pick_out_dishes + 
      (total_dishes - oliver_edible_dishes - mango_salsa_dishes - mango_jelly_dishes) :=
by sorry

end buffet_dishes_l3982_398282


namespace cupcake_combinations_l3982_398203

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of cupcakes to be purchased -/
def total_cupcakes : ℕ := 7

/-- The number of cupcake types available -/
def cupcake_types : ℕ := 5

/-- The number of cupcake types that must have at least one selected -/
def required_types : ℕ := 4

/-- The number of remaining cupcakes after selecting one of each required type -/
def remaining_cupcakes : ℕ := total_cupcakes - required_types

theorem cupcake_combinations : 
  stars_and_bars remaining_cupcakes cupcake_types = 35 := by
  sorry

end cupcake_combinations_l3982_398203


namespace john_good_games_l3982_398210

/-- 
Given:
- John bought 21 games from a friend
- John bought 8 games at a garage sale
- 23 of the games didn't work

Prove that John ended up with 6 good games.
-/
theorem john_good_games : 
  let games_from_friend : ℕ := 21
  let games_from_garage_sale : ℕ := 8
  let non_working_games : ℕ := 23
  let total_games := games_from_friend + games_from_garage_sale
  let good_games := total_games - non_working_games
  good_games = 6 := by
  sorry

end john_good_games_l3982_398210


namespace average_equation_solution_l3982_398268

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 69 → a = 26 := by
  sorry

end average_equation_solution_l3982_398268


namespace polynomial_shift_root_existence_l3982_398204

/-- A polynomial of degree 10 with leading coefficient 1 -/
def Polynomial10 := {p : Polynomial ℝ // p.degree = 10 ∧ p.leadingCoeff = 1}

theorem polynomial_shift_root_existence (P Q : Polynomial10) 
  (h : ∀ x : ℝ, P.val.eval x ≠ Q.val.eval x) :
  ∃ x : ℝ, (P.val.eval (x + 1)) = (Q.val.eval (x - 1)) := by
  sorry

end polynomial_shift_root_existence_l3982_398204


namespace border_area_l3982_398202

def photo_height : ℝ := 9
def photo_width : ℝ := 12
def frame_border : ℝ := 3

theorem border_area : 
  let framed_height := photo_height + 2 * frame_border
  let framed_width := photo_width + 2 * frame_border
  let photo_area := photo_height * photo_width
  let framed_area := framed_height * framed_width
  framed_area - photo_area = 162 := by sorry

end border_area_l3982_398202


namespace quadratic_function_comparison_l3982_398262

/-- Proves that for points A(x₁, y₁) and B(x₂, y₂) on the graph of y = (x - 1)² + 1, 
    if x₁ > x₂ > 1, then y₁ > y₂. -/
theorem quadratic_function_comparison (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = (x₁ - 1)^2 + 1)
    (h2 : y₂ = (x₂ - 1)^2 + 1)
    (h3 : x₁ > x₂)
    (h4 : x₂ > 1) : 
  y₁ > y₂ := by
  sorry


end quadratic_function_comparison_l3982_398262


namespace log_27_3_l3982_398233

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  have h : 27 = 3^3 := by sorry
  sorry

end log_27_3_l3982_398233


namespace not_multiple_of_three_l3982_398257

theorem not_multiple_of_three (n : ℕ) (h : ∃ m : ℕ, n * (n + 3) = m ^ 2) : ¬ (3 ∣ n) := by
  sorry

end not_multiple_of_three_l3982_398257


namespace circle_center_coordinate_difference_l3982_398286

/-- Given two points as endpoints of a circle's diameter, 
    calculate the absolute difference between the x and y coordinates of the circle's center -/
theorem circle_center_coordinate_difference (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = 8 ∧ y₁ = -7)
  (h2 : x₂ = -4 ∧ y₂ = 5) : 
  |((x₁ + x₂) / 2) - ((y₁ + y₂) / 2)| = 3 := by
  sorry

end circle_center_coordinate_difference_l3982_398286


namespace limit_inequality_l3982_398213

theorem limit_inequality : 12.37 * (3/2 - 1/3) > Real.cos (π/10) := by
  sorry

end limit_inequality_l3982_398213


namespace student_gathering_problem_l3982_398293

theorem student_gathering_problem (male_count : ℕ) (female_count : ℕ) : 
  female_count = male_count + 6 →
  (female_count : ℚ) / (male_count + female_count) = 2 / 3 →
  male_count + female_count = 18 :=
by sorry

end student_gathering_problem_l3982_398293


namespace bathtub_jello_cost_l3982_398221

/-- The cost to fill a bathtub with jello given specific ratios and measurements -/
theorem bathtub_jello_cost :
  let jello_mix_per_pound : ℚ := 3/2  -- 1.5 tablespoons per pound
  let bathtub_volume : ℚ := 6         -- 6 cubic feet
  let gallons_per_cubic_foot : ℚ := 15/2  -- 7.5 gallons per cubic foot
  let pounds_per_gallon : ℚ := 8      -- 8 pounds per gallon
  let cost_per_tablespoon : ℚ := 1/2  -- $0.50 per tablespoon
  
  let total_gallons : ℚ := bathtub_volume * gallons_per_cubic_foot
  let total_pounds : ℚ := total_gallons * pounds_per_gallon
  let total_tablespoons : ℚ := total_pounds * jello_mix_per_pound
  let total_cost : ℚ := total_tablespoons * cost_per_tablespoon

  total_cost = 270 := by
    sorry


end bathtub_jello_cost_l3982_398221


namespace h_zero_at_seven_fifths_l3982_398281

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 7

-- Theorem statement
theorem h_zero_at_seven_fifths : h (7 / 5) = 0 := by
  sorry

end h_zero_at_seven_fifths_l3982_398281


namespace school_attendance_problem_l3982_398207

theorem school_attendance_problem (girls : ℕ) (percentage_increase : ℚ) (boys : ℕ) :
  girls = 5000 →
  percentage_increase = 40 / 100 →
  (boys : ℚ) + percentage_increase * (boys : ℚ) = (boys : ℚ) + (girls : ℚ) →
  boys = 12500 := by
sorry

end school_attendance_problem_l3982_398207


namespace ball_hit_ground_time_l3982_398254

/-- The time when a ball hits the ground given its height equation -/
theorem ball_hit_ground_time (t : ℝ) : 
  let y : ℝ → ℝ := λ t => -4.9 * t^2 + 4 * t + 6
  y t = 0 → t = 78 / 49 := by
sorry

end ball_hit_ground_time_l3982_398254


namespace cube_rotation_theorem_l3982_398280

/-- Represents the orientation of a picture on the top face of a cube -/
inductive PictureOrientation
| Original
| Rotated90
| Rotated180

/-- Represents a cube with a picture on its top face -/
structure Cube :=
  (orientation : PictureOrientation)

/-- Represents the action of rolling a cube across its edges -/
def roll (c : Cube) : Cube :=
  sorry

/-- Represents a sequence of rolls that returns the cube to its original position -/
def rollSequence (c : Cube) : Cube :=
  sorry

theorem cube_rotation_theorem (c : Cube) :
  (∃ (seq : Cube → Cube), seq c = Cube.mk PictureOrientation.Rotated180) ∧
  (∀ (seq : Cube → Cube), seq c ≠ Cube.mk PictureOrientation.Rotated90) :=
sorry

end cube_rotation_theorem_l3982_398280


namespace divisibility_theorem_l3982_398296

theorem divisibility_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end divisibility_theorem_l3982_398296


namespace negative_number_identification_l3982_398248

theorem negative_number_identification :
  (|-2023| ≥ 0) ∧ 
  (Real.sqrt ((-2)^2) ≥ 0) ∧ 
  (0 ≥ 0) ∧ 
  (-3^2 < 0) := by
  sorry

end negative_number_identification_l3982_398248


namespace boys_circle_distance_l3982_398218

/-- The least total distance traveled by 8 boys on a circle -/
theorem boys_circle_distance (n : ℕ) (r : ℝ) (h_n : n = 8) (h_r : r = 30) :
  let chord_length := 2 * r * Real.sqrt ((2 : ℝ) + Real.sqrt 2) / 2
  let non_adjacent_count := n - 3
  let total_distance := n * non_adjacent_count * chord_length
  total_distance = 1200 * Real.sqrt ((2 : ℝ) + Real.sqrt 2) :=
by sorry

end boys_circle_distance_l3982_398218


namespace amp_four_two_l3982_398278

-- Define the & operation
def amp (a b : ℝ) : ℝ := ((a + b) * (a - b))^2

-- Theorem statement
theorem amp_four_two : amp 4 2 = 144 := by
  sorry

end amp_four_two_l3982_398278


namespace probability_red_ball_experiment_l3982_398251

/-- The probability of picking a red ball in an experiment -/
def probability_red_ball (total_experiments : ℕ) (red_picks : ℕ) : ℚ :=
  red_picks / total_experiments

/-- Theorem: Given 10 experiments where red balls were picked 4 times, 
    the probability of picking a red ball is 0.4 -/
theorem probability_red_ball_experiment : 
  probability_red_ball 10 4 = 0.4 := by
  sorry

end probability_red_ball_experiment_l3982_398251


namespace sum_of_squares_geq_product_l3982_398255

theorem sum_of_squares_geq_product (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ x₁ * (x₂ + x₃ + x₄ + x₅) := by
  sorry

end sum_of_squares_geq_product_l3982_398255


namespace percentage_of_games_sold_l3982_398267

theorem percentage_of_games_sold (initial_cost : ℝ) (sold_price : ℝ) : 
  initial_cost = 200 → 
  sold_price = 240 → 
  (sold_price / (initial_cost * 3)) * 100 = 40 := by
  sorry

end percentage_of_games_sold_l3982_398267


namespace parallel_vectors_y_value_l3982_398227

theorem parallel_vectors_y_value (a b : ℝ × ℝ) :
  a = (6, 2) →
  b.2 = 3 →
  (∃ k : ℝ, k ≠ 0 ∧ a = k • b) →
  b.1 = 9 := by
  sorry

end parallel_vectors_y_value_l3982_398227


namespace A_intersect_B_l3982_398284

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}

theorem A_intersect_B : A ∩ B = {2} := by sorry

end A_intersect_B_l3982_398284


namespace greatest_two_digit_with_product_12_and_smallest_sum_l3982_398261

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem greatest_two_digit_with_product_12_and_smallest_sum :
  ∃ (n : ℕ), is_two_digit n ∧ 
             digit_product n = 12 ∧
             (∀ m : ℕ, is_two_digit m → digit_product m = 12 → digit_sum m ≥ digit_sum n) ∧
             (∀ k : ℕ, is_two_digit k → digit_product k = 12 → digit_sum k = digit_sum n → k ≤ n) ∧
             n = 43 :=
by sorry

end greatest_two_digit_with_product_12_and_smallest_sum_l3982_398261


namespace hexagonal_pyramid_base_edge_length_l3982_398294

/-- A pyramid with a regular hexagon base -/
structure HexagonalPyramid where
  base_edge_length : ℝ
  side_edge_length : ℝ
  total_edge_length : ℝ

/-- The property that the pyramid satisfies the given conditions -/
def satisfies_conditions (p : HexagonalPyramid) : Prop :=
  p.side_edge_length = 8 ∧ p.total_edge_length = 120

/-- The theorem stating that if a hexagonal pyramid satisfies the conditions, 
    its base edge length is 12 -/
theorem hexagonal_pyramid_base_edge_length 
  (p : HexagonalPyramid) (h : satisfies_conditions p) : 
  p.base_edge_length = 12 := by
  sorry

end hexagonal_pyramid_base_edge_length_l3982_398294


namespace existence_of_always_different_teams_l3982_398283

/-- Represents a team assignment for a single game -/
def GameAssignment := Fin 22 → Bool

/-- Represents the team assignments for all three games -/
def ThreeGamesAssignment := Fin 3 → GameAssignment

theorem existence_of_always_different_teams (games : ThreeGamesAssignment) : 
  ∃ (p1 p2 : Fin 22), p1 ≠ p2 ∧ 
    (∀ (g : Fin 3), games g p1 ≠ games g p2) :=
sorry

end existence_of_always_different_teams_l3982_398283


namespace sqrt_18_times_sqrt_32_l3982_398299

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_18_times_sqrt_32_l3982_398299
