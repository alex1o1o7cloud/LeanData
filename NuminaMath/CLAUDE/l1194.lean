import Mathlib

namespace NUMINAMATH_CALUDE_carpenter_table_difference_carpenter_table_difference_proof_l1194_119450

theorem carpenter_table_difference : ℕ → ℕ → ℕ → Prop :=
  fun this_month total difference =>
    this_month = 10 →
    total = 17 →
    difference = this_month - (total - this_month) →
    difference = 3

-- The proof is omitted
theorem carpenter_table_difference_proof : carpenter_table_difference 10 17 3 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_table_difference_carpenter_table_difference_proof_l1194_119450


namespace NUMINAMATH_CALUDE_jellybean_count_l1194_119495

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- The number of jellybeans Caleb has -/
def caleb_jellybeans : ℕ := 3 * dozen

/-- The number of jellybeans Sophie has -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The number of jellybeans Max has -/
def max_jellybeans : ℕ := sophie_jellybeans + 2 * dozen

/-- The total number of jellybeans -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans + max_jellybeans

theorem jellybean_count : total_jellybeans = 96 := by sorry

end NUMINAMATH_CALUDE_jellybean_count_l1194_119495


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1194_119445

/-- Given a function f(x) = 2a^x + 3, where a > 0 and a ≠ 1, prove that f(0) = 5 -/
theorem function_passes_through_point
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (f : ℝ → ℝ) 
  (h3 : ∀ x, f x = 2 * a^x + 3) : 
  f 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1194_119445


namespace NUMINAMATH_CALUDE_suit_price_calculation_l1194_119426

theorem suit_price_calculation (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 200 →
  increase_rate = 0.3 →
  discount_rate = 0.3 →
  original_price * (1 + increase_rate) * (1 - discount_rate) = 182 := by
  sorry

#check suit_price_calculation

end NUMINAMATH_CALUDE_suit_price_calculation_l1194_119426


namespace NUMINAMATH_CALUDE_median_of_consecutive_integers_l1194_119463

theorem median_of_consecutive_integers (n : ℕ) (a : ℤ) (h : n > 0) :
  (∀ i, 0 ≤ i ∧ i < n → (a + i) + (a + (n - 1) - i) = 120) →
  (n % 2 = 1 → (a + (n - 1) / 2) = 60) ∧
  (n % 2 = 0 → (2 * a + n - 1) / 2 = 60) :=
sorry

end NUMINAMATH_CALUDE_median_of_consecutive_integers_l1194_119463


namespace NUMINAMATH_CALUDE_work_trip_speed_l1194_119449

/-- Proves that given a round trip of 3 hours, where the return journey takes 1.2 hours at 120 km/h,
    and the journey to work takes 1.8 hours, the average speed to work is 80 km/h. -/
theorem work_trip_speed (total_time : ℝ) (return_time : ℝ) (return_speed : ℝ) (to_work_time : ℝ)
    (h1 : total_time = 3)
    (h2 : return_time = 1.2)
    (h3 : return_speed = 120)
    (h4 : to_work_time = 1.8)
    (h5 : total_time = return_time + to_work_time) :
    (return_speed * return_time) / to_work_time = 80 := by
  sorry

end NUMINAMATH_CALUDE_work_trip_speed_l1194_119449


namespace NUMINAMATH_CALUDE_multiplier_is_three_l1194_119469

theorem multiplier_is_three (x y a : ℤ) : 
  a * x + y = 40 →
  2 * x - y = 20 →
  3 * y^2 = 48 →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l1194_119469


namespace NUMINAMATH_CALUDE_t_shape_perimeter_is_20_l1194_119489

/-- The perimeter of a T shape formed by two rectangles -/
def t_shape_perimeter (width height overlap : ℝ) : ℝ :=
  2 * (width + height) + 2 * (height - 2 * overlap)

/-- Theorem: The perimeter of the T shape is 20 inches -/
theorem t_shape_perimeter_is_20 :
  t_shape_perimeter 3 5 1.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_is_20_l1194_119489


namespace NUMINAMATH_CALUDE_pool_and_deck_area_l1194_119447

/-- Calculates the total area of a rectangular pool and its surrounding deck. -/
def total_area (pool_length pool_width deck_width : ℝ) : ℝ :=
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width)

/-- Proves that the total area of a specific rectangular pool and its deck is 728 square feet. -/
theorem pool_and_deck_area :
  total_area 20 22 3 = 728 := by
  sorry

end NUMINAMATH_CALUDE_pool_and_deck_area_l1194_119447


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1194_119436

/-- A polynomial with positive real coefficients -/
structure PositivePolynomial where
  coeffs : List ℝ
  all_positive : ∀ c ∈ coeffs, c > 0

/-- Evaluate a polynomial at a given point -/
def evalPoly (p : PositivePolynomial) (x : ℝ) : ℝ :=
  p.coeffs.enum.foldl (λ acc (i, a) => acc + a * x ^ i) 0

/-- The main theorem -/
theorem polynomial_inequality (p : PositivePolynomial) :
  (evalPoly p 1 ≥ 1 / evalPoly p 1) →
  (∀ x : ℝ, x > 0 → evalPoly p (1/x) ≥ 1 / evalPoly p x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1194_119436


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1194_119464

theorem absolute_value_simplification (a : ℝ) (h : a < 3) : |a - 3| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1194_119464


namespace NUMINAMATH_CALUDE_draining_time_is_independent_variable_l1194_119424

/-- Represents the water volume in the reservoir --/
def water_volume (t : ℝ) : ℝ := 50 - 2 * t

theorem draining_time_is_independent_variable :
  ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → water_volume t₁ ≠ water_volume t₂ :=
by sorry

end NUMINAMATH_CALUDE_draining_time_is_independent_variable_l1194_119424


namespace NUMINAMATH_CALUDE_sticks_per_hour_to_stay_warm_l1194_119412

/-- The number of sticks of wood produced by chopping up furniture -/
def sticks_per_furniture : Nat → Nat
| 0 => 6  -- chairs
| 1 => 9  -- tables
| 2 => 2  -- stools
| _ => 0  -- other furniture (not considered)

/-- The number of each type of furniture Mary chopped up -/
def furniture_count : Nat → Nat
| 0 => 18  -- chairs
| 1 => 6   -- tables
| 2 => 4   -- stools
| _ => 0   -- other furniture (not considered)

/-- The number of hours Mary can keep warm -/
def warm_hours : Nat := 34

/-- Calculates the total number of sticks of wood Mary has -/
def total_sticks : Nat :=
  (sticks_per_furniture 0 * furniture_count 0) +
  (sticks_per_furniture 1 * furniture_count 1) +
  (sticks_per_furniture 2 * furniture_count 2)

/-- The theorem to prove -/
theorem sticks_per_hour_to_stay_warm :
  total_sticks / warm_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_sticks_per_hour_to_stay_warm_l1194_119412


namespace NUMINAMATH_CALUDE_bus_children_difference_l1194_119467

theorem bus_children_difference (initial : ℕ) (got_off : ℕ) (final : ℕ) : 
  initial = 5 → got_off = 63 → final = 14 → 
  ∃ (got_on : ℕ), got_on - got_off = 9 ∧ initial - got_off + got_on = final :=
sorry

end NUMINAMATH_CALUDE_bus_children_difference_l1194_119467


namespace NUMINAMATH_CALUDE_sequence_nth_term_l1194_119422

theorem sequence_nth_term (u : ℕ → ℝ) (u₀ a b : ℝ) (h : ∀ n : ℕ, u (n + 1) = a * u n + b) :
  ∀ n : ℕ, u n = if a = 1
    then u₀ + n * b
    else a^n * u₀ + b * (1 - a^(n + 1)) / (1 - a) :=
by sorry

end NUMINAMATH_CALUDE_sequence_nth_term_l1194_119422


namespace NUMINAMATH_CALUDE_pentagon_cannot_tessellate_l1194_119439

/-- A regular polygon can tessellate a plane if its internal angle divides 360° evenly -/
def can_tessellate (internal_angle : ℝ) : Prop :=
  ∃ n : ℕ, n * internal_angle = 360

/-- The internal angle of a regular pentagon is 108° -/
def pentagon_internal_angle : ℝ := 108

/-- Theorem: A regular pentagon cannot tessellate a plane by itself -/
theorem pentagon_cannot_tessellate :
  ¬(can_tessellate pentagon_internal_angle) :=
sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tessellate_l1194_119439


namespace NUMINAMATH_CALUDE_line_problem_l1194_119443

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  let slope1 := (l1.point2.2 - l1.point1.2) / (l1.point2.1 - l1.point1.1)
  let slope2 := (l2.point2.2 - l2.point1.2) / (l2.point2.1 - l2.point1.1)
  slope1 = slope2

/-- Check if a line intersects another line defined by ax + by = c -/
def intersects (l : Line) (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (y - l.point1.2) / (x - l.point1.1) = (l.point2.2 - l.point1.2) / (l.point2.1 - l.point1.1) ∧
    a * x + b * y = c

/-- The main theorem -/
theorem line_problem (k : ℝ) : 
  let l1 := Line.mk (3, -7) (k, 20)
  let l2 := Line.mk (-28/5, 7) (0, 7)
  are_parallel l1 l2 ∧ 
  intersects l1 1 (-3) 5 →
  k = -18.6 := by
sorry

end NUMINAMATH_CALUDE_line_problem_l1194_119443


namespace NUMINAMATH_CALUDE_sum_of_factors_l1194_119430

theorem sum_of_factors (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -120 →
  p + q + r + s + t = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1194_119430


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1194_119486

theorem area_between_concentric_circles
  (r_small : ℝ) (r_large : ℝ) (h1 : r_small * 2 = 6)
  (h2 : r_large = 3 * r_small) :
  π * r_large^2 - π * r_small^2 = 72 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1194_119486


namespace NUMINAMATH_CALUDE_lcm_1362_918_l1194_119413

theorem lcm_1362_918 : Nat.lcm 1362 918 = 69462 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1362_918_l1194_119413


namespace NUMINAMATH_CALUDE_factor_polynomial_l1194_119496

theorem factor_polynomial (x : ℝ) : 
  x^2 + 6*x + 9 - 16*x^4 = (-4*x^2 + 2*x + 3)*(4*x^2 + 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1194_119496


namespace NUMINAMATH_CALUDE_print_shop_copies_l1194_119416

theorem print_shop_copies (x_price y_price difference : ℚ) (h1 : x_price = 1.25)
  (h2 : y_price = 2.75) (h3 : difference = 90) :
  ∃ n : ℚ, n * y_price = n * x_price + difference ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_copies_l1194_119416


namespace NUMINAMATH_CALUDE_group_size_l1194_119427

/-- The number of members in the group -/
def n : ℕ := sorry

/-- The total collection in paise -/
def total_paise : ℕ := 5776

/-- Each member contributes as many paise as there are members -/
axiom member_contribution : n = total_paise / n

theorem group_size : n = 76 := by sorry

end NUMINAMATH_CALUDE_group_size_l1194_119427


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1194_119404

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1194_119404


namespace NUMINAMATH_CALUDE_log_product_change_base_l1194_119433

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_product_change_base 
  (a b c : ℝ) (m n : ℝ) 
  (h1 : log b a = m) 
  (h2 : log c b = n) 
  (h3 : a > 0) (h4 : b > 1) (h5 : c > 1) :
  log (b * c) (a * b) = n * (m + 1) / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_log_product_change_base_l1194_119433


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l1194_119423

/-- Represents a rectangular shape with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of tiles that can fit in one dimension -/
def fitInDimension (floorSize tileSize : ℕ) : ℕ :=
  floorSize / tileSize

/-- Calculates the number of tiles that can fit on the floor for a given orientation -/
def tilesForOrientation (floor tile : Rectangle) : ℕ :=
  (fitInDimension floor.width tile.width) * (fitInDimension floor.height tile.height)

/-- Theorem: The maximum number of 50x40 tiles on a 120x150 floor is 9 -/
theorem max_tiles_on_floor :
  let floor : Rectangle := ⟨120, 150⟩
  let tile : Rectangle := ⟨50, 40⟩
  let orientation1 := tilesForOrientation floor tile
  let orientation2 := tilesForOrientation floor ⟨tile.height, tile.width⟩
  max orientation1 orientation2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l1194_119423


namespace NUMINAMATH_CALUDE_student_mistake_fraction_l1194_119462

theorem student_mistake_fraction (original_number : ℕ) 
  (h1 : original_number = 384) 
  (correct_fraction : ℚ) 
  (h2 : correct_fraction = 5 / 16) 
  (mistake_fraction : ℚ) : 
  (mistake_fraction * original_number = correct_fraction * original_number + 200) → 
  mistake_fraction = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_student_mistake_fraction_l1194_119462


namespace NUMINAMATH_CALUDE_cone_volume_l1194_119408

/-- The volume of a cone with base radius 1 and unfolded side surface area 2π -/
theorem cone_volume (r : Real) (side_area : Real) (h : Real) : 
  r = 1 → side_area = 2 * Real.pi → h = Real.sqrt 3 → 
  (1 / 3 : Real) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 := by
  sorry

#check cone_volume

end NUMINAMATH_CALUDE_cone_volume_l1194_119408


namespace NUMINAMATH_CALUDE_cricket_team_size_l1194_119493

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  n : ℕ  -- number of team members
  captain_age : ℕ
  wicket_keeper_age : ℕ
  team_avg_age : ℚ
  remaining_avg_age : ℚ

/-- The cricket team satisfies the given conditions -/
def satisfies_conditions (team : CricketTeam) : Prop :=
  team.captain_age = 27 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.team_avg_age = 24 ∧
  team.remaining_avg_age = team.team_avg_age - 1 ∧
  team.n * team.team_avg_age = team.captain_age + team.wicket_keeper_age + (team.n - 2) * team.remaining_avg_age

/-- The number of members in the cricket team that satisfies the conditions is 11 -/
theorem cricket_team_size :
  ∃ (team : CricketTeam), satisfies_conditions team ∧ team.n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l1194_119493


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l1194_119429

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age1 : ℕ) (leaving_age2 : ℕ) (remaining_people : ℕ) :
  initial_people = 6 →
  initial_average = 25 →
  leaving_age1 = 20 →
  leaving_age2 = 22 →
  remaining_people = 4 →
  (initial_people : ℚ) * initial_average - (leaving_age1 + leaving_age2 : ℚ) = 
    (remaining_people : ℚ) * 27 := by
  sorry

#check average_age_after_leaving

end NUMINAMATH_CALUDE_average_age_after_leaving_l1194_119429


namespace NUMINAMATH_CALUDE_runners_journey_l1194_119410

/-- A runner's journey with changing speeds -/
theorem runners_journey (initial_speed : ℝ) (tired_speed : ℝ) (total_distance : ℝ) (total_time : ℝ)
  (h1 : initial_speed = 15)
  (h2 : tired_speed = 10)
  (h3 : total_distance = 100)
  (h4 : total_time = 9) :
  ∃ (initial_time : ℝ), initial_time = 2 ∧ 
    initial_speed * initial_time + tired_speed * (total_time - initial_time) = total_distance := by
  sorry

end NUMINAMATH_CALUDE_runners_journey_l1194_119410


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l1194_119483

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_series : ℕ → ℕ
  | 0 => 0
  | n + 1 => sum_series n + if (n + 1) % 3 = 0 ∧ n + 1 ≤ 9 then 2 * factorial (n + 1) else 0

theorem last_two_digits_of_sum : last_two_digits (sum_series 99) = 24 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_l1194_119483


namespace NUMINAMATH_CALUDE_union_of_sets_l1194_119407

theorem union_of_sets (S R : Set ℕ) : 
  S = {1} → R = {1, 2} → S ∪ R = {1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l1194_119407


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1194_119417

/-- Proves that the ratio of father's age to son's age is 4:1 given the conditions -/
theorem father_son_age_ratio :
  ∀ (father_age son_age : ℕ),
    father_age = 64 →
    son_age = 16 →
    father_age - 10 + son_age - 10 = 60 →
    father_age / son_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1194_119417


namespace NUMINAMATH_CALUDE_equation_solution_l1194_119484

theorem equation_solution : 
  ∃ x : ℝ, (3 / (2 * x + 1) = 5 / (4 * x)) ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1194_119484


namespace NUMINAMATH_CALUDE_trick_deck_cost_l1194_119497

theorem trick_deck_cost (frank_decks : ℕ) (friend_decks : ℕ) (total_spent : ℕ) :
  frank_decks = 3 →
  friend_decks = 2 →
  total_spent = 35 →
  ∃ (cost : ℕ), frank_decks * cost + friend_decks * cost = total_spent ∧ cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_l1194_119497


namespace NUMINAMATH_CALUDE_original_number_proof_l1194_119452

theorem original_number_proof : 
  ∃! x : ℕ, (x + 4) % 23 = 0 ∧ ∀ y : ℕ, y < 4 → (x + y) % 23 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1194_119452


namespace NUMINAMATH_CALUDE_middle_integer_of_pairwise_sums_l1194_119437

theorem middle_integer_of_pairwise_sums (x y z : ℤ) 
  (h1 : x < y) (h2 : y < z)
  (sum_xy : x + y = 22)
  (sum_xz : x + z = 24)
  (sum_yz : y + z = 16) :
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_middle_integer_of_pairwise_sums_l1194_119437


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1194_119420

/-- The distance between the vertices of the hyperbola x^2/144 - y^2/49 = 1 is 24 -/
theorem hyperbola_vertex_distance : 
  let a : ℝ := Real.sqrt 144
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 144 - y^2 / 49 = 1
  2 * a = 24 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1194_119420


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1194_119457

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 81)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_proportional : ∃ (k : ℚ), a = 3 * k ∧ b = 5 * k ∧ c = 7 * k)
  (h_sum : a + b + c = total) :
  a = 81 / 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1194_119457


namespace NUMINAMATH_CALUDE_room_puzzle_solution_l1194_119499

/-- Represents a person who can either be a truth-teller or a liar -/
inductive Person
| TruthTeller
| Liar

/-- Represents a statement made by a person -/
structure Statement where
  content : Prop
  speaker : Person

/-- The environment of the problem -/
structure Room where
  people : Nat
  liars : Nat
  statements : List Statement

/-- The correct solution to the problem -/
def correct_solution : Room := { people := 4, liars := 2, statements := [] }

/-- Checks if a given solution is consistent with the statements made -/
def is_consistent (room : Room) : Prop :=
  let s1 := Statement.mk (room.people ≤ 3 ∧ room.liars = room.people) Person.Liar
  let s2 := Statement.mk (room.people ≤ 4 ∧ room.liars < room.people) Person.TruthTeller
  let s3 := Statement.mk (room.people = 5 ∧ room.liars = 3) Person.Liar
  room.statements = [s1, s2, s3]

/-- The main theorem to prove -/
theorem room_puzzle_solution :
  ∀ room : Room, is_consistent room → room = correct_solution :=
sorry

end NUMINAMATH_CALUDE_room_puzzle_solution_l1194_119499


namespace NUMINAMATH_CALUDE_range_of_function_l1194_119438

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ y = x + 4 / x) → y ≤ -4 ∨ y ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l1194_119438


namespace NUMINAMATH_CALUDE_square_diff_product_l1194_119434

theorem square_diff_product (m n : ℝ) (h1 : m - n = 4) (h2 : m * n = -3) :
  (m^2 - 4) * (n^2 - 4) = -15 := by sorry

end NUMINAMATH_CALUDE_square_diff_product_l1194_119434


namespace NUMINAMATH_CALUDE_negation_of_true_is_false_l1194_119446

theorem negation_of_true_is_false (p : Prop) : p → ¬p = False := by
  sorry

end NUMINAMATH_CALUDE_negation_of_true_is_false_l1194_119446


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1194_119490

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1194_119490


namespace NUMINAMATH_CALUDE_brick_surface_area_l1194_119444

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 2 cm rectangular prism is 136 cm² -/
theorem brick_surface_area :
  surface_area 10 4 2 = 136 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1194_119444


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l1194_119478

/-- The number of chickens -/
def num_chickens : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The number of cats -/
def num_cats : ℕ := 5

/-- The number of empty cages -/
def num_empty_cages : ℕ := 2

/-- The total number of entities (animals + empty cages) -/
def total_entities : ℕ := num_chickens + num_dogs + num_cats + num_empty_cages

/-- The number of animal groups -/
def num_groups : ℕ := 3

/-- The number of possible positions for empty cages -/
def num_positions : ℕ := num_groups + 2

theorem happy_valley_kennel_arrangement :
  (Nat.factorial num_groups) * (Nat.choose num_positions num_empty_cages) *
  (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats) = 1036800 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l1194_119478


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_three_first_quadrant_iff_m_lt_neg_two_or_gt_three_l1194_119479

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

-- Theorem 1: z is a pure imaginary number iff m = 3
theorem pure_imaginary_iff_m_eq_three (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = 3 := by sorry

-- Theorem 2: z is in the first quadrant iff m < -2 or m > 3
theorem first_quadrant_iff_m_lt_neg_two_or_gt_three (m : ℝ) :
  (z m).re > 0 ∧ (z m).im > 0 ↔ m < -2 ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_three_first_quadrant_iff_m_lt_neg_two_or_gt_three_l1194_119479


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1194_119473

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 54 →
  volume = (surface_area / 6) * (surface_area / 6).sqrt →
  volume = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1194_119473


namespace NUMINAMATH_CALUDE_peter_and_susan_money_l1194_119472

/-- The total amount of money Peter and Susan have together -/
def total_money (peter_amount susan_amount : ℚ) : ℚ :=
  peter_amount + susan_amount

/-- Theorem stating that Peter and Susan have 0.65 dollars altogether -/
theorem peter_and_susan_money :
  total_money (2/5) (1/4) = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_peter_and_susan_money_l1194_119472


namespace NUMINAMATH_CALUDE_min_difference_triangle_sides_l1194_119466

theorem min_difference_triangle_sides (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2010 →
  PQ < QR →
  QR < PR →
  (∀ PQ' QR' PR' : ℕ, 
    PQ' + QR' + PR' = 2010 →
    PQ' < QR' →
    QR' < PR' →
    QR - PQ ≤ QR' - PQ') →
  QR - PQ = 1 := by
sorry

end NUMINAMATH_CALUDE_min_difference_triangle_sides_l1194_119466


namespace NUMINAMATH_CALUDE_total_trees_on_farm_l1194_119465

def farm_trees (mango_trees : ℕ) (coconut_trees : ℕ) : ℕ :=
  mango_trees + coconut_trees

theorem total_trees_on_farm :
  let mango_trees : ℕ := 60
  let coconut_trees : ℕ := mango_trees / 2 - 5
  farm_trees mango_trees coconut_trees = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_on_farm_l1194_119465


namespace NUMINAMATH_CALUDE_quadratic_negative_roots_l1194_119476

theorem quadratic_negative_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m+2)*x + m + 5 = 0 → x < 0) ↔ m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_roots_l1194_119476


namespace NUMINAMATH_CALUDE_one_third_of_recipe_l1194_119432

theorem one_third_of_recipe (original_amount : ℚ) (reduced_amount : ℚ) : 
  original_amount = 27/4 → reduced_amount = original_amount / 3 → reduced_amount = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_one_third_of_recipe_l1194_119432


namespace NUMINAMATH_CALUDE_three_digit_number_appended_l1194_119448

theorem three_digit_number_appended (n : ℕ) : 
  100 ≤ n ∧ n < 1000 → 1000 * n + n = 1001 * n := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_appended_l1194_119448


namespace NUMINAMATH_CALUDE_equation_solution_l1194_119415

theorem equation_solution : ∃ x : ℝ, 9 - x - 2 * (31 - x) = 27 ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1194_119415


namespace NUMINAMATH_CALUDE_negation_equivalence_l1194_119440

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x^2 - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1194_119440


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1194_119425

theorem cyclic_sum_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1194_119425


namespace NUMINAMATH_CALUDE_petrol_price_reduction_l1194_119456

/-- The original price of petrol in dollars per gallon -/
def P : ℝ := sorry

/-- The amount spent on petrol in dollars -/
def amount_spent : ℝ := 250

/-- The price reduction percentage as a decimal -/
def price_reduction : ℝ := 0.1

/-- The additional gallons that can be bought after the price reduction -/
def additional_gallons : ℝ := 5

/-- Theorem stating the relationship between the original price and the additional gallons that can be bought after the price reduction -/
theorem petrol_price_reduction (P : ℝ) (amount_spent : ℝ) (price_reduction : ℝ) (additional_gallons : ℝ) :
  amount_spent / ((1 - price_reduction) * P) - amount_spent / P = additional_gallons :=
sorry

end NUMINAMATH_CALUDE_petrol_price_reduction_l1194_119456


namespace NUMINAMATH_CALUDE_equation_solution_l1194_119454

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-25 * 2 + 50)| ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1194_119454


namespace NUMINAMATH_CALUDE_triangle_6_8_10_is_right_l1194_119459

-- Define a triangle with sides a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

-- Theorem: A triangle with sides 6, 8, and 10 is a right triangle
theorem triangle_6_8_10_is_right : 
  let t : Triangle := { a := 6, b := 8, c := 10 }
  isRightTriangle t := by
  sorry


end NUMINAMATH_CALUDE_triangle_6_8_10_is_right_l1194_119459


namespace NUMINAMATH_CALUDE_oranges_bought_l1194_119451

/-- Represents the fruit shopping scenario over a week -/
structure FruitShopping where
  apples : ℕ
  oranges : ℕ
  total_fruits : apples + oranges = 5
  total_cost : ℕ
  cost_is_whole_dollars : total_cost % 100 = 0
  cost_calculation : total_cost = 30 * apples + 45 * oranges + 20

/-- Theorem stating that the number of oranges bought is 2 -/
theorem oranges_bought (shop : FruitShopping) : shop.oranges = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_bought_l1194_119451


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l1194_119453

theorem simplify_sqrt_fraction : 
  (Real.sqrt 45) / (2 * Real.sqrt 20) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l1194_119453


namespace NUMINAMATH_CALUDE_eagles_volleyball_games_l1194_119455

theorem eagles_volleyball_games :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
  initial_wins = (0.4 : ℝ) * initial_games →
  (initial_wins + 9 : ℝ) / (initial_games + 10) = 0.55 →
  initial_games + 10 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_eagles_volleyball_games_l1194_119455


namespace NUMINAMATH_CALUDE_max_quadratic_expression_l1194_119477

theorem max_quadratic_expression :
  ∃ (M : ℝ), M = 67 ∧ ∀ (p : ℝ), -3 * p^2 + 30 * p - 8 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_quadratic_expression_l1194_119477


namespace NUMINAMATH_CALUDE_triangle_properties_l1194_119487

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.b * Real.sin t.A = 3 * t.c * Real.sin t.B ∧
  t.a = 3 ∧
  Real.cos t.B = 2/3

theorem triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.b = Real.sqrt 6 ∧ 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1194_119487


namespace NUMINAMATH_CALUDE_somu_age_relation_somu_age_relation_past_somu_present_age_l1194_119485

/-- Somu's present age -/
def somu_age : ℕ := sorry

/-- Somu's father's present age -/
def father_age : ℕ := sorry

/-- Theorem stating the relationship between Somu's age and his father's age -/
theorem somu_age_relation : somu_age = father_age / 3 := by sorry

/-- Theorem stating the relationship between Somu's and his father's ages 10 years ago -/
theorem somu_age_relation_past : somu_age - 10 = (father_age - 10) / 5 := by sorry

/-- Main theorem proving Somu's present age -/
theorem somu_present_age : somu_age = 20 := by sorry

end NUMINAMATH_CALUDE_somu_age_relation_somu_age_relation_past_somu_present_age_l1194_119485


namespace NUMINAMATH_CALUDE_expression_value_l1194_119458

theorem expression_value (x y : ℝ) (h : |x + 1| + (y - 2)^2 = 0) :
  4 * x^2 * y - (6 * x * y - 3 * (4 * x * y - 2) - x^2 * y) + 1 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1194_119458


namespace NUMINAMATH_CALUDE_bashers_win_probability_l1194_119474

def probability_at_least_4_out_of_5 (p : ℝ) : ℝ :=
  5 * p^4 * (1 - p) + p^5

theorem bashers_win_probability :
  probability_at_least_4_out_of_5 (4/5) = 3072/3125 := by
  sorry

end NUMINAMATH_CALUDE_bashers_win_probability_l1194_119474


namespace NUMINAMATH_CALUDE_student_failed_marks_l1194_119494

def total_marks : ℕ := 400
def passing_percentage : ℚ := 45 / 100
def obtained_marks : ℕ := 150

theorem student_failed_marks :
  (total_marks * passing_percentage).floor - obtained_marks = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_failed_marks_l1194_119494


namespace NUMINAMATH_CALUDE_area_at_stage_8_l1194_119401

/-- Calculates the number of squares added up to a given stage -/
def squaresAdded (stage : ℕ) : ℕ :=
  (stage + 1) / 2

/-- The side length of each square in inches -/
def squareSideLength : ℕ := 4

/-- Calculates the area of the figure at a given stage -/
def areaAtStage (stage : ℕ) : ℕ :=
  (squaresAdded stage) * (squareSideLength * squareSideLength)

/-- Proves that the area of the figure at Stage 8 is 64 square inches -/
theorem area_at_stage_8 : areaAtStage 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1194_119401


namespace NUMINAMATH_CALUDE_tangent_line_ln_curve_l1194_119470

/-- The equation of the tangent line to y = ln(x+1) at (1, ln 2) -/
theorem tangent_line_ln_curve (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.log (t + 1)
  let tangent_point : ℝ × ℝ := (1, Real.log 2)
  let tangent_line : ℝ → ℝ → Prop := λ a b => x - 2*y - 1 + 2*(Real.log 2) = 0
  (∀ t, (t, f t) ∈ Set.range (λ u => (u, f u))) →  -- curve condition
  tangent_point.1 = 1 ∧ tangent_point.2 = Real.log 2 → -- point condition
  (∃ k : ℝ, ∀ a b, tangent_line a b ↔ b - tangent_point.2 = k * (a - tangent_point.1)) -- tangent line property
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_ln_curve_l1194_119470


namespace NUMINAMATH_CALUDE_office_call_probabilities_l1194_119468

/-- Represents the probability of a call being for a specific person -/
structure CallProbability where
  A : ℚ
  B : ℚ
  C : ℚ
  sum_to_one : A + B + C = 1

/-- Calculates the probability of all three calls being for the same person -/
def prob_all_same (p : CallProbability) : ℚ :=
  p.A^3 + p.B^3 + p.C^3

/-- Calculates the probability of exactly two out of three calls being for A -/
def prob_two_for_A (p : CallProbability) : ℚ :=
  3 * p.A^2 * (1 - p.A)

theorem office_call_probabilities :
  ∃ (p : CallProbability),
    p.A = 1/6 ∧ p.B = 1/3 ∧ p.C = 1/2 ∧
    prob_all_same p = 1/6 ∧
    prob_two_for_A p = 5/72 := by
  sorry

end NUMINAMATH_CALUDE_office_call_probabilities_l1194_119468


namespace NUMINAMATH_CALUDE_equation_solution_l1194_119428

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (12 + 3*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1194_119428


namespace NUMINAMATH_CALUDE_sum_of_even_integers_202_to_300_l1194_119471

def sum_of_first_n_even_integers (n : ℕ) : ℕ := n * (n + 1)

def count_even_numbers_in_range (first last : ℕ) : ℕ :=
  (last - first) / 2 + 1

def sum_of_arithmetic_sequence (n first last : ℕ) : ℕ :=
  n / 2 * (first + last)

theorem sum_of_even_integers_202_to_300 
  (h : sum_of_first_n_even_integers 50 = 2550) :
  sum_of_arithmetic_sequence 
    (count_even_numbers_in_range 202 300) 
    202 
    300 = 12550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_202_to_300_l1194_119471


namespace NUMINAMATH_CALUDE_babysitting_earnings_l1194_119482

/-- Calculates the amount earned for a given hour of babysitting -/
def hourlyRate (hour : ℕ) : ℕ :=
  2 * (hour % 6 + 1)

/-- Calculates the total amount earned for a given number of hours -/
def totalEarned (hours : ℕ) : ℕ :=
  (List.range hours).map hourlyRate |>.sum

theorem babysitting_earnings : totalEarned 48 = 288 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l1194_119482


namespace NUMINAMATH_CALUDE_second_box_clay_capacity_l1194_119488

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.height * d.width * d.length

/-- The dimensions of the first box -/
def firstBox : BoxDimensions := {
  height := 3,
  width := 4,
  length := 7
}

/-- The dimensions of the second box -/
def secondBox : BoxDimensions := {
  height := 3 * firstBox.height,
  width := 2 * firstBox.width,
  length := firstBox.length
}

/-- The amount of clay the first box can hold in grams -/
def firstBoxClay : ℝ := 70

/-- Theorem: The second box can hold 420 grams of clay -/
theorem second_box_clay_capacity : 
  (boxVolume secondBox / boxVolume firstBox) * firstBoxClay = 420 := by sorry

end NUMINAMATH_CALUDE_second_box_clay_capacity_l1194_119488


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_estate_area_l1194_119480

/-- Represents the scale of the map --/
def map_scale : ℚ := 500 / 2

/-- Represents the length of the diagonals on the map in inches --/
def diagonal_length : ℚ := 10

/-- Calculates the actual length of the diagonal in miles --/
def actual_diagonal_length : ℚ := diagonal_length * map_scale

/-- Represents an isosceles trapezoid estate --/
structure IsoscelesTrapezoidEstate where
  diagonal : ℚ
  area : ℚ

/-- Theorem stating the area of the isosceles trapezoid estate --/
theorem isosceles_trapezoid_estate_area :
  ∃ (estate : IsoscelesTrapezoidEstate),
    estate.diagonal = actual_diagonal_length ∧
    estate.area = 3125000 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_trapezoid_estate_area_l1194_119480


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l1194_119460

theorem danny_bottle_caps (initial : ℕ) (found : ℕ) (current : ℕ) (thrown_away : ℕ) : 
  initial = 69 → found = 58 → current = 67 → 
  thrown_away = initial + found - current →
  thrown_away = 60 := by
sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l1194_119460


namespace NUMINAMATH_CALUDE_find_number_l1194_119435

theorem find_number : ∃ x : ℝ, 4.75 + 0.303 + x = 5.485 ∧ x = 0.432 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1194_119435


namespace NUMINAMATH_CALUDE_unique_fixed_point_l1194_119461

-- Define the type for points in the plane
variable (Point : Type)

-- Define the type for lines in the plane
variable (Line : Type)

-- Define the set of all lines in the plane
variable (L : Set Line)

-- Define the function f that assigns a point to each line
variable (f : Line → Point)

-- Define a predicate to check if a point is on a line
variable (on_line : Point → Line → Prop)

-- Define a predicate to check if points are on a circle
variable (on_circle : Point → Point → Point → Point → Prop)

-- Axiom: f(l) is on l for all lines l
axiom f_on_line : ∀ l : Line, on_line (f l) l

-- Axiom: For any point X and any three lines l1, l2, l3 passing through X,
--        the points f(l1), f(l2), f(l3), and X lie on a circle
axiom circle_property : 
  ∀ (X : Point) (l1 l2 l3 : Line),
  on_line X l1 → on_line X l2 → on_line X l3 →
  on_circle X (f l1) (f l2) (f l3)

-- Theorem: There exists a unique point P such that f(l) = P for any line l passing through P
theorem unique_fixed_point :
  ∃! P : Point, ∀ l : Line, on_line P l → f l = P :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_l1194_119461


namespace NUMINAMATH_CALUDE_geometric_sum_ratio_l1194_119491

/-- Given a geometric sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- The ratio of S_5 to S_10 is 1/3 -/
axiom ratio_condition : S 5 / S 10 = 1 / 3

/-- Theorem: If S_5 / S_10 = 1/3, then S_5 / (S_20 + S_10) = 1/18 -/
theorem geometric_sum_ratio : S 5 / (S 20 + S 10) = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_ratio_l1194_119491


namespace NUMINAMATH_CALUDE_x_seventh_plus_27x_squared_l1194_119431

theorem x_seventh_plus_27x_squared (x : ℝ) (h : x^3 - 3*x = 7) :
  x^7 + 27*x^2 = 76*x^2 + 270*x + 483 := by
  sorry

end NUMINAMATH_CALUDE_x_seventh_plus_27x_squared_l1194_119431


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1194_119481

/-- Given a trapezoid PQRS with specified dimensions, prove the length of QR -/
theorem trapezoid_side_length (area : ℝ) (altitude PQ RS : ℝ) (h1 : area = 210)
  (h2 : altitude = 10) (h3 : PQ = 12) (h4 : RS = 21) :
  ∃ QR : ℝ, QR = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l1194_119481


namespace NUMINAMATH_CALUDE_tomato_price_theorem_l1194_119421

/-- The original price per pound of tomatoes -/
def original_price : ℝ := 0.80

/-- The percentage of tomatoes that can be sold -/
def sellable_percentage : ℝ := 0.90

/-- The selling price per pound of tomatoes -/
def selling_price : ℝ := 0.96

/-- The profit percentage of the cost -/
def profit_percentage : ℝ := 0.08

/-- Theorem stating that the original price satisfies the given conditions -/
theorem tomato_price_theorem :
  selling_price * sellable_percentage = 
  original_price * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_tomato_price_theorem_l1194_119421


namespace NUMINAMATH_CALUDE_triangle_max_area_l1194_119419

/-- Given a triangle ABC with side a = √2 and acosB + bsinA = c, 
    the maximum area of the triangle is (√2 + 1) / 2 -/
theorem triangle_max_area (a b c A B C : Real) :
  a = Real.sqrt 2 →
  a * Real.cos B + b * Real.sin A = c →
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
    S ≤ (Real.sqrt 2 + 1) / 2 ∧
    (S = (Real.sqrt 2 + 1) / 2 ↔ b = c)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1194_119419


namespace NUMINAMATH_CALUDE_solve_marbles_problem_l1194_119411

def marbles_problem (wolfgang_marbles : ℕ) : Prop :=
  let ludo_marbles := wolfgang_marbles + (wolfgang_marbles / 4)
  let total_wolfgang_ludo := wolfgang_marbles + ludo_marbles
  let michael_marbles := (2 * total_wolfgang_ludo) / 3
  let total_marbles := total_wolfgang_ludo + michael_marbles
  (wolfgang_marbles = 16) →
  (total_marbles / 3 = 20)

theorem solve_marbles_problem :
  marbles_problem 16 := by sorry

end NUMINAMATH_CALUDE_solve_marbles_problem_l1194_119411


namespace NUMINAMATH_CALUDE_solve_for_b_l1194_119405

theorem solve_for_b (b : ℚ) (h : b + b/4 = 10/4) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1194_119405


namespace NUMINAMATH_CALUDE_pen_and_notebook_cost_l1194_119414

theorem pen_and_notebook_cost (pen_cost : ℝ) (price_difference : ℝ) : 
  pen_cost = 4.5 → price_difference = 1.8 → pen_cost + (pen_cost - price_difference) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_pen_and_notebook_cost_l1194_119414


namespace NUMINAMATH_CALUDE_original_number_proof_l1194_119418

theorem original_number_proof :
  ∀ x : ℕ,
  x < 10 →
  (x + 10) * ((x + 10) / x) = 72 →
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1194_119418


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1194_119403

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + (3*x + 6)) / 5 = 30 → x = 96/7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1194_119403


namespace NUMINAMATH_CALUDE_eccentricity_difference_l1194_119442

/-- Eccentricity function for a hyperbola -/
def eccentricity_function (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (θ : ℝ) : ℝ :=
  sorry

/-- Theorem: Difference of eccentricities for specific angles -/
theorem eccentricity_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  eccentricity_function a b h1 h2 (2 * Real.pi / 3) - eccentricity_function a b h1 h2 (Real.pi / 3) = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_difference_l1194_119442


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_F_powers_of_two_l1194_119406

-- Define the function F recursively
def F : ℕ → ℚ
  | 0 => 0
  | 1 => 3/2
  | (n+2) => 5/2 * F (n+1) - F n

-- Define the series
def series_sum : ℕ → ℚ
  | 0 => 1 / F (2^0)
  | (n+1) => series_sum n + 1 / F (2^(n+1))

-- State the theorem
theorem sum_of_reciprocal_F_powers_of_two :
  ∃ (L : ℚ), L = 1 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |series_sum n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_F_powers_of_two_l1194_119406


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l1194_119498

/-- The range of values for the real number a such that at least one of the given quadratic equations has real roots -/
theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + a^2 = 0 ∨ x^2 + 2*a*x - 2*a = 0) ↔ 
  a ≤ -2 ∨ (-1/3 ≤ a ∧ a < 1) ∨ 0 ≤ a := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l1194_119498


namespace NUMINAMATH_CALUDE_apples_per_pie_l1194_119492

theorem apples_per_pie (initial_apples : Nat) (handed_out : Nat) (num_pies : Nat)
  (h1 : initial_apples = 96)
  (h2 : handed_out = 42)
  (h3 : num_pies = 9) :
  (initial_apples - handed_out) / num_pies = 6 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l1194_119492


namespace NUMINAMATH_CALUDE_g_is_zero_l1194_119475

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (Real.sin x ^ 4 + 3 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end NUMINAMATH_CALUDE_g_is_zero_l1194_119475


namespace NUMINAMATH_CALUDE_largest_non_expressible_l1194_119409

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 36 * a + b ∧ a > 0 ∧ is_power_of_two b

theorem largest_non_expressible : 
  (∀ n > 104, expressible n) ∧ ¬(expressible 104) := by sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l1194_119409


namespace NUMINAMATH_CALUDE_max_roses_proof_l1194_119402

-- Define the pricing structure
def individual_price : ℚ := 6.3
def dozen_price : ℚ := 36
def two_dozen_price : ℚ := 50
def five_dozen_price : ℚ := 110

-- Define Maria's budget constraints
def total_budget : ℚ := 680
def min_red_roses_budget : ℚ := 200

-- Define the function to calculate the maximum number of roses
def max_roses : ℕ := 360

-- Theorem statement
theorem max_roses_proof :
  ∀ (purchase_strategy : ℕ → ℕ → ℕ → ℕ → ℚ),
  (∀ a b c d, purchase_strategy a b c d * individual_price +
              purchase_strategy a b c d * dozen_price / 12 +
              purchase_strategy a b c d * two_dozen_price / 24 +
              purchase_strategy a b c d * five_dozen_price / 60 ≤ total_budget) →
  (∀ a b c d, purchase_strategy a b c d * five_dozen_price / 60 ≥ min_red_roses_budget) →
  (∀ a b c d, purchase_strategy a b c d + purchase_strategy a b c d * 12 +
              purchase_strategy a b c d * 24 + purchase_strategy a b c d * 60 ≤ max_roses) :=
by sorry

end NUMINAMATH_CALUDE_max_roses_proof_l1194_119402


namespace NUMINAMATH_CALUDE_cindy_calculation_l1194_119400

def original_number : ℝ := (23 * 5) + 7

theorem cindy_calculation : 
  Int.floor ((original_number + 7) / 5) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l1194_119400


namespace NUMINAMATH_CALUDE_min_dominoes_to_win_viktors_winning_strategy_l1194_119441

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a domino placement on the board -/
structure DominoPlacement :=
  (board : Board)
  (num_dominoes : ℕ)

/-- Theorem: The minimum number of dominoes Viktor needs to fix to win -/
theorem min_dominoes_to_win (b : Board) (d : DominoPlacement) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem viktors_winning_strategy (b : Board) (d : DominoPlacement) : 
  b.size = 2022 → d.board = b → d.num_dominoes = 2022 * 2022 / 2 → 
  min_dominoes_to_win b d = 1011^2 :=
sorry

end NUMINAMATH_CALUDE_min_dominoes_to_win_viktors_winning_strategy_l1194_119441
