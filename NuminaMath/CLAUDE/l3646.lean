import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3646_364678

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 40, and 36, the distance between two adjacent parallel lines is 7.8. -/
theorem parallel_lines_distance (r : ℝ) : 
  let d := (4336 : ℝ) / 71
  (40 : ℝ) * r^2 = 16000 + 10 * d ∧ 
  (36 : ℝ) * r^2 = 11664 + 81 * d → 
  Real.sqrt d = 7.8 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3646_364678


namespace NUMINAMATH_CALUDE_det_A_nonzero_l3646_364683

def matrix_A (n : ℕ) (a : ℤ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => a^(i.val * j.val + 1)

theorem det_A_nonzero {n : ℕ} {a : ℤ} (h : a > 1) :
  Matrix.det (matrix_A n a) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_det_A_nonzero_l3646_364683


namespace NUMINAMATH_CALUDE_caleb_dandelion_friends_l3646_364667

/-- The number of friends Caleb shared dandelion puffs with -/
def num_friends (total : ℕ) (mom sister grandma dog friend : ℕ) : ℕ :=
  (total - (mom + sister + grandma + dog)) / friend

/-- Theorem stating the number of friends Caleb shared dandelion puffs with -/
theorem caleb_dandelion_friends :
  num_friends 40 3 3 5 2 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_caleb_dandelion_friends_l3646_364667


namespace NUMINAMATH_CALUDE_combined_stickers_l3646_364611

/-- The combined total of cat stickers for June and Bonnie after receiving gifts from their grandparents -/
theorem combined_stickers (june_initial : ℕ) (bonnie_initial : ℕ) (gift : ℕ) 
  (h1 : june_initial = 76)
  (h2 : bonnie_initial = 63)
  (h3 : gift = 25) :
  june_initial + bonnie_initial + 2 * gift = 189 := by
  sorry

#check combined_stickers

end NUMINAMATH_CALUDE_combined_stickers_l3646_364611


namespace NUMINAMATH_CALUDE_sqrt_product_l3646_364613

theorem sqrt_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l3646_364613


namespace NUMINAMATH_CALUDE_min_bailing_rate_calculation_l3646_364665

-- Define the problem parameters
def distance_to_shore : Real := 2 -- miles
def water_intake_rate : Real := 15 -- gallons per minute
def max_water_capacity : Real := 60 -- gallons
def rowing_speed : Real := 3 -- miles per hour

-- Define the theorem
theorem min_bailing_rate_calculation :
  let time_to_shore := distance_to_shore / rowing_speed * 60 -- Convert to minutes
  let total_water_intake := water_intake_rate * time_to_shore
  let water_to_bail := total_water_intake - max_water_capacity
  let min_bailing_rate := water_to_bail / time_to_shore
  min_bailing_rate = 13.5 := by
  sorry


end NUMINAMATH_CALUDE_min_bailing_rate_calculation_l3646_364665


namespace NUMINAMATH_CALUDE_pine_tree_branches_l3646_364694

/-- The number of branches in a pine tree -/
def num_branches : ℕ := 23

/-- The movements of the squirrel from the middle branch to the top -/
def movements : List ℤ := [5, -7, 4, 9]

/-- The number of branches from the middle to the top -/
def branches_to_top : ℕ := (movements.sum).toNat

theorem pine_tree_branches :
  num_branches = 2 * branches_to_top + 1 :=
by sorry

end NUMINAMATH_CALUDE_pine_tree_branches_l3646_364694


namespace NUMINAMATH_CALUDE_max_profit_at_price_l3646_364672

/-- Represents the daily sales and profit model of a store --/
structure StoreModel where
  cost_price : ℝ
  max_price_factor : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ

/-- The store model satisfies the given conditions --/
def satisfies_conditions (model : StoreModel) : Prop :=
  model.cost_price = 100 ∧
  model.max_price_factor = 1.4 ∧
  model.sales_function 130 = 140 ∧
  model.sales_function 140 = 120 ∧
  (∀ x, model.sales_function x = -2 * x + 400) ∧
  (∀ x, model.profit_function x = (x - model.cost_price) * model.sales_function x)

/-- The maximum profit occurs at the given price and value --/
theorem max_profit_at_price (model : StoreModel) 
    (h : satisfies_conditions model) :
    (∀ x, x ≤ model.max_price_factor * model.cost_price → 
      model.profit_function x ≤ model.profit_function 140) ∧
    model.profit_function 140 = 4800 := by
  sorry

#check max_profit_at_price

end NUMINAMATH_CALUDE_max_profit_at_price_l3646_364672


namespace NUMINAMATH_CALUDE_candy_left_l3646_364660

theorem candy_left (initial : ℝ) (morning : ℝ) (afternoon : ℝ) :
  initial = 38 →
  morning = 7.5 →
  afternoon = 15.25 →
  initial - morning - afternoon = 15.25 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_left_l3646_364660


namespace NUMINAMATH_CALUDE_oliver_seashell_difference_l3646_364644

/-- The number of seashells Oliver collected on Monday -/
def monday_shells : ℕ := 2

/-- The total number of seashells Oliver collected -/
def total_shells : ℕ := 4

/-- The number of seashells Oliver collected on Tuesday -/
def tuesday_shells : ℕ := total_shells - monday_shells

/-- Theorem: Oliver collected 2 more seashells on Tuesday compared to Monday -/
theorem oliver_seashell_difference : tuesday_shells - monday_shells = 2 := by
  sorry

end NUMINAMATH_CALUDE_oliver_seashell_difference_l3646_364644


namespace NUMINAMATH_CALUDE_min_value_theorem_l3646_364676

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 400 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3646_364676


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3646_364668

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ),
    (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
      5 * x^2 / ((x - 4) * (x - 2)^3) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^3) ∧
    P = 10 ∧ Q = -10 ∧ R = -10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3646_364668


namespace NUMINAMATH_CALUDE_seven_by_seven_grid_shaded_percentage_l3646_364628

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : ℕ)

/-- Calculates the percentage of shaded area in a square grid -/
def shadedPercentage (grid : SquareGrid) : ℚ :=
  (grid.shaded : ℚ) / (grid.size * grid.size : ℚ) * 100

/-- Theorem: The percentage of shaded area in a 7x7 grid with 7 shaded squares is (1/7) * 100% -/
theorem seven_by_seven_grid_shaded_percentage :
  let grid : SquareGrid := ⟨7, 7⟩
  shadedPercentage grid = 100 / 7 := by sorry

end NUMINAMATH_CALUDE_seven_by_seven_grid_shaded_percentage_l3646_364628


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3646_364631

/-- Proves that a rectangular field with length 7/5 times its width and perimeter 432 meters has a width of 90 meters -/
theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 432 →
  perimeter = 2 * length + 2 * width →
  width = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3646_364631


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l3646_364614

/-- A geometric sequence {a_n} satisfying given conditions -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (k : ℕ), a (k + 1) = q * a k

theorem geometric_sequence_solution (a : ℕ → ℚ) :
  geometric_sequence a →
  a 3 + a 6 = 36 →
  a 4 + a 7 = 18 →
  (∃ n : ℕ, a n = 1/2) →
  ∃ n : ℕ, a n = 1/2 ∧ n = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l3646_364614


namespace NUMINAMATH_CALUDE_clapping_groups_l3646_364634

def number_of_people : ℕ := 4043
def claps_per_hand : ℕ := 2021

def valid_groups (n k : ℕ) : ℕ := Nat.choose n k

def invalid_groups (n m : ℕ) : ℕ := n * Nat.choose m 2

theorem clapping_groups :
  valid_groups number_of_people 3 - invalid_groups number_of_people claps_per_hand =
  valid_groups number_of_people 3 - number_of_people * valid_groups claps_per_hand 2 :=
by sorry

end NUMINAMATH_CALUDE_clapping_groups_l3646_364634


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3646_364616

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let bounceSequence := (List.range numBounces).map (fun n => initialHeight * bounceRatio^n)
  let ascents := bounceSequence.sum
  let descents := initialHeight + (List.take (numBounces - 1) bounceSequence).sum
  ascents + descents

/-- The problem statement -/
theorem ball_bounce_distance :
  ∃ (d : ℝ), abs (totalDistance 20 (2/3) 4 - d) < 1 ∧ Int.floor d = 68 := by
  sorry


end NUMINAMATH_CALUDE_ball_bounce_distance_l3646_364616


namespace NUMINAMATH_CALUDE_mean_home_runs_l3646_364637

def home_runs : List (Nat × Nat) := [(5, 5), (9, 3), (7, 4), (11, 2)]

theorem mean_home_runs :
  let total_home_runs := (home_runs.map (λ (hr, players) => hr * players)).sum
  let total_players := (home_runs.map (λ (_, players) => players)).sum
  (total_home_runs : ℚ) / total_players = 729/100 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3646_364637


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l3646_364630

/-- Calculate the profit from a lemonade stand -/
theorem lemonade_stand_profit 
  (price_per_cup : ℕ) 
  (cups_sold : ℕ) 
  (lemon_cost sugar_cost cup_cost : ℕ) : 
  price_per_cup * cups_sold - (lemon_cost + sugar_cost + cup_cost) = 66 :=
by
  sorry

#check lemonade_stand_profit 4 21 10 5 3

end NUMINAMATH_CALUDE_lemonade_stand_profit_l3646_364630


namespace NUMINAMATH_CALUDE_mirror_area_is_2016_l3646_364689

/-- Calculates the area of a rectangular mirror inside a frame with rounded corners. -/
def mirror_area (frame_width : ℝ) (frame_height : ℝ) (frame_side_width : ℝ) : ℝ :=
  (frame_width - 2 * frame_side_width) * (frame_height - 2 * frame_side_width)

/-- Proves that the area of the mirror is 2016 cm² given the frame dimensions. -/
theorem mirror_area_is_2016 :
  mirror_area 50 70 7 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_is_2016_l3646_364689


namespace NUMINAMATH_CALUDE_water_amount_l3646_364624

/-- Represents the recipe ratios and quantities -/
structure Recipe where
  water : ℝ
  sugar : ℝ
  cranberry : ℝ
  water_sugar_ratio : water = 5 * sugar
  sugar_cranberry_ratio : sugar = 3 * cranberry
  cranberry_amount : cranberry = 4

/-- Proves that the amount of water needed is 60 cups -/
theorem water_amount (r : Recipe) : r.water = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_l3646_364624


namespace NUMINAMATH_CALUDE_probability_s30_to_40th_position_l3646_364649

def bubble_pass (s : List ℝ) : List ℝ := sorry

theorem probability_s30_to_40th_position 
  (s : List ℝ) 
  (h1 : s.length = 41) 
  (h2 : s.Nodup) 
  : ℝ := by
  sorry

#check probability_s30_to_40th_position

end NUMINAMATH_CALUDE_probability_s30_to_40th_position_l3646_364649


namespace NUMINAMATH_CALUDE_amount_division_l3646_364696

/-- Given an amount divided into 3 parts proportional to 1/2 : 2/3 : 3/4, 
    with the first part being 204, prove the total amount is 782. -/
theorem amount_division (amount : ℕ) 
  (h1 : amount > 0)
  (h2 : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a * 3 * 4 = 1 * 2 * 4 ∧ 
    b * 2 * 4 = 2 * 3 * 4 ∧ 
    c * 2 * 3 = 3 * 2 * 4 ∧
    a + b + c = amount)
  (h3 : a = 204) : 
  amount = 782 := by
  sorry

end NUMINAMATH_CALUDE_amount_division_l3646_364696


namespace NUMINAMATH_CALUDE_sequence_sum_l3646_364648

theorem sequence_sum (n : ℕ) (x : ℕ → ℚ) : 
  (∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + 1 / 3) →
  x 1 = 2 →
  n > 0 →
  Finset.sum (Finset.range n) (λ i => x (i + 1)) = n * (n + 11) / 6 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_l3646_364648


namespace NUMINAMATH_CALUDE_gas_pressure_calculation_l3646_364652

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  inv_prop : volume * pressure = volume * pressure

/-- The initial state of the gas -/
def initial_state : GasState where
  volume := 3
  pressure := 8
  inv_prop := by sorry

/-- The final state of the gas -/
def final_state : GasState where
  volume := 7.5
  pressure := 3.2
  inv_prop := by sorry

/-- Theorem stating that the final pressure is correct given the initial conditions -/
theorem gas_pressure_calculation (initial : GasState) (final : GasState)
    (h_initial : initial = initial_state)
    (h_final_volume : final.volume = 7.5)
    (h_const : initial.volume * initial.pressure = final.volume * final.pressure) :
    final.pressure = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_gas_pressure_calculation_l3646_364652


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l3646_364662

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if points are coplanar
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to count the number of unique planes determined by four points
def countPlanesFromPoints (p1 p2 p3 p4 : Point3D) : ℕ := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ areCoplanar p1 p2 p3 p4) : 
  countPlanesFromPoints p1 p2 p3 p4 = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l3646_364662


namespace NUMINAMATH_CALUDE_empty_set_is_proposition_l3646_364604

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (truth_value : Bool), (s = "true") ∨ (s = "false")

-- The statement we want to prove is a proposition
def empty_set_statement : String := "The empty set is a subset of any set"

-- Theorem statement
theorem empty_set_is_proposition : is_proposition empty_set_statement := by
  sorry


end NUMINAMATH_CALUDE_empty_set_is_proposition_l3646_364604


namespace NUMINAMATH_CALUDE_parameterized_line_solution_l3646_364677

/-- The line y = 4x - 7 parameterized by (x, y) = (s, -3) + t(3, m) -/
def parameterized_line (s m t : ℝ) : ℝ × ℝ :=
  (s + 3*t, -3 + m*t)

/-- The line y = 4x - 7 -/
def line (x y : ℝ) : Prop :=
  y = 4*x - 7

theorem parameterized_line_solution :
  ∃ (s m : ℝ), ∀ (t : ℝ),
    let (x, y) := parameterized_line s m t
    line x y ∧ s = 1 ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_parameterized_line_solution_l3646_364677


namespace NUMINAMATH_CALUDE_image_and_preimage_of_f_l3646_364670

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem image_and_preimage_of_f :
  (f (3, 5) = (-2, 8)) ∧ (f (4, 1) = (-2, 8)) := by sorry

end NUMINAMATH_CALUDE_image_and_preimage_of_f_l3646_364670


namespace NUMINAMATH_CALUDE_problem_statement_l3646_364666

theorem problem_statement (a b c : ℝ) : 
  a^2 + b^2 + c^2 + 4 ≤ a*b + 3*b + 2*c → 200*a + 9*b + c = 219 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3646_364666


namespace NUMINAMATH_CALUDE_billys_songbook_l3646_364619

/-- The number of songs Billy can play -/
def songs_can_play : ℕ := 24

/-- The number of songs Billy still needs to learn -/
def songs_to_learn : ℕ := 28

/-- The total number of songs in Billy's music book -/
def total_songs : ℕ := songs_can_play + songs_to_learn

theorem billys_songbook :
  total_songs = 52 := by sorry

end NUMINAMATH_CALUDE_billys_songbook_l3646_364619


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3646_364698

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 60 → B = 2 * C → A + B + C = 180 → B = 80 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3646_364698


namespace NUMINAMATH_CALUDE_pet_beds_per_pet_l3646_364618

theorem pet_beds_per_pet (total_beds : ℕ) (num_pets : ℕ) (beds_per_pet : ℕ) : 
  total_beds = 20 → num_pets = 10 → beds_per_pet = total_beds / num_pets → beds_per_pet = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_beds_per_pet_l3646_364618


namespace NUMINAMATH_CALUDE_intersection_equals_open_unit_interval_l3646_364682

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define the open interval (0, 1)
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_equals_open_unit_interval : M ∩ N = open_unit_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_unit_interval_l3646_364682


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l3646_364691

/-- The number of ways to arrange 9 objects, where there are 3 objects of each of 3 types -/
def arrangement_count : ℕ := 1680

/-- The total number of objects -/
def total_objects : ℕ := 9

/-- The number of different types of objects -/
def num_types : ℕ := 3

/-- The number of objects of each type -/
def objects_per_type : ℕ := 3

theorem arrangement_count_proof :
  arrangement_count = (Nat.factorial total_objects) / 
    (Nat.factorial objects_per_type ^ num_types) :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l3646_364691


namespace NUMINAMATH_CALUDE_anna_final_collection_l3646_364622

structure StampCollection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)

def initial_anna : StampCollection := ⟨10, 15, 12⟩
def initial_alison : StampCollection := ⟨8, 10, 10⟩
def initial_jeff : StampCollection := ⟨12, 9, 10⟩

def transaction1 (anna alison : StampCollection) : StampCollection :=
  ⟨anna.nature + alison.nature / 2, anna.architecture + alison.architecture / 2, anna.animals + alison.animals / 2⟩

def transaction2 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature + 2, anna.architecture, anna.animals - 1⟩

def transaction3 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature, anna.architecture + 3, anna.animals - 5⟩

def transaction4 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature + 7, anna.architecture, anna.animals - 4⟩

def final_anna : StampCollection :=
  transaction4 (transaction3 (transaction2 (transaction1 initial_anna initial_alison)))

theorem anna_final_collection :
  final_anna = ⟨23, 23, 7⟩ := by sorry

end NUMINAMATH_CALUDE_anna_final_collection_l3646_364622


namespace NUMINAMATH_CALUDE_no_five_coprime_two_digit_composites_l3646_364650

theorem no_five_coprime_two_digit_composites : 
  ¬ (∃ (a b c d e : ℕ), 
    (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (10 ≤ c ∧ c ≤ 99) ∧ (10 ≤ d ∧ d ≤ 99) ∧ (10 ≤ e ∧ e ≤ 99) ∧
    (¬ Nat.Prime a) ∧ (¬ Nat.Prime b) ∧ (¬ Nat.Prime c) ∧ (¬ Nat.Prime d) ∧ (¬ Nat.Prime e) ∧
    (Nat.gcd a b = 1) ∧ (Nat.gcd a c = 1) ∧ (Nat.gcd a d = 1) ∧ (Nat.gcd a e = 1) ∧
    (Nat.gcd b c = 1) ∧ (Nat.gcd b d = 1) ∧ (Nat.gcd b e = 1) ∧
    (Nat.gcd c d = 1) ∧ (Nat.gcd c e = 1) ∧
    (Nat.gcd d e = 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_five_coprime_two_digit_composites_l3646_364650


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3646_364629

theorem cube_volume_from_surface_area :
  ∀ (surface_area volume : ℝ),
  surface_area = 384 →
  (surface_area / 6).sqrt ^ 3 = volume →
  volume = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3646_364629


namespace NUMINAMATH_CALUDE_factorial_not_ending_1976_zeros_l3646_364663

theorem factorial_not_ending_1976_zeros (n : ℕ) : ∃ k : ℕ, n! % (10^k) ≠ 1976 * (10^k) :=
sorry

end NUMINAMATH_CALUDE_factorial_not_ending_1976_zeros_l3646_364663


namespace NUMINAMATH_CALUDE_total_pages_read_l3646_364635

-- Define the book's properties
def total_pages : ℕ := 95
def total_chapters : ℕ := 8

-- Define Jake's reading
def initial_pages_read : ℕ := 37
def additional_pages_read : ℕ := 25

-- Theorem to prove
theorem total_pages_read :
  initial_pages_read + additional_pages_read = 62 :=
by sorry

end NUMINAMATH_CALUDE_total_pages_read_l3646_364635


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l3646_364612

theorem modular_inverse_of_5_mod_31 : ∃ x : ℕ, x ∈ Finset.range 31 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l3646_364612


namespace NUMINAMATH_CALUDE_apples_given_correct_l3646_364692

/-- The number of apples the farmer originally had -/
def original_apples : ℕ := 127

/-- The number of apples the farmer now has -/
def current_apples : ℕ := 39

/-- The number of apples given to the neighbor -/
def apples_given : ℕ := original_apples - current_apples

theorem apples_given_correct : apples_given = 88 := by sorry

end NUMINAMATH_CALUDE_apples_given_correct_l3646_364692


namespace NUMINAMATH_CALUDE_angles_equal_necessary_not_sufficient_l3646_364606

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the angle between a line and a plane
variable (angle : Line → Plane → ℝ)

-- State the theorem
theorem angles_equal_necessary_not_sufficient
  (m n : Line) (a : Plane) :
  (∀ (l₁ l₂ : Line), parallel l₁ l₂ → angle l₁ a = angle l₂ a) ∧
  ¬(∀ (l₁ l₂ : Line), angle l₁ a = angle l₂ a → parallel l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_angles_equal_necessary_not_sufficient_l3646_364606


namespace NUMINAMATH_CALUDE_trig_expression_equals_half_l3646_364685

/-- Proves that the given trigonometric expression equals 1/2 --/
theorem trig_expression_equals_half : 
  (Real.sin (70 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_half_l3646_364685


namespace NUMINAMATH_CALUDE_special_circle_equation_l3646_364643

/-- A circle with center on y = 2x and specific chord lengths -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 2 * center.1
  x_chord_length : 4 = 2 * (radius ^ 2 - center.1 ^ 2).sqrt
  y_chord_length : 8 = 2 * (radius ^ 2 - center.2 ^ 2).sqrt

/-- The equation of the circle is one of two specific forms -/
theorem special_circle_equation (c : SpecialCircle) :
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 5 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) ∨
  (∀ x y : ℝ, (x + 1) ^ 2 + (y + 2) ^ 2 = 5 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l3646_364643


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3646_364632

theorem geometric_sequence_ratio (q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  q = 1/2 →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, a (n + 1) = a n * q) →
  S 4 / a 3 = 15/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3646_364632


namespace NUMINAMATH_CALUDE_circle_F_value_l3646_364601

-- Define the circle equation
def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_F_value :
  ∃ F : ℝ, (∀ x y : ℝ, circle_equation x y F ↔ (x - 1)^2 + (y + 1)^2 = circle_radius^2) ∧ F = -2 :=
sorry

end NUMINAMATH_CALUDE_circle_F_value_l3646_364601


namespace NUMINAMATH_CALUDE_monkey_liar_puzzle_l3646_364642

-- Define the possible characteristics
inductive Character
| Monkey
| NonMonkey

inductive Truthfulness
| TruthTeller
| Liar

-- Define a structure for an individual
structure Individual where
  species : Character
  honesty : Truthfulness

-- Define the statements made by A and B
def statement_A (a b : Individual) : Prop :=
  a.species = Character.Monkey ∧ b.species = Character.Monkey

def statement_B (a b : Individual) : Prop :=
  a.honesty = Truthfulness.Liar ∧ b.honesty = Truthfulness.Liar

-- Theorem stating the solution
theorem monkey_liar_puzzle :
  ∃ (a b : Individual),
    (statement_A a b ↔ a.honesty = Truthfulness.TruthTeller) ∧
    (statement_B a b ↔ b.honesty = Truthfulness.Liar) ∧
    a.species = Character.Monkey ∧
    b.species = Character.Monkey ∧
    a.honesty = Truthfulness.TruthTeller ∧
    b.honesty = Truthfulness.Liar :=
  sorry


end NUMINAMATH_CALUDE_monkey_liar_puzzle_l3646_364642


namespace NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_is_one_l3646_364639

/-- The probability that at least 4 people stay for the entire game, given that
    8 people come to a basketball game, 4 are certain to stay, and 4 have a
    1/3 probability of staying. -/
theorem basketball_game_probability : Real :=
  let total_people : ℕ := 8
  let certain_stayers : ℕ := 4
  let uncertain_stayers : ℕ := 4
  let stay_probability : Real := 1/3
  1

/-- Proof that the probability is indeed 1. -/
theorem basketball_game_probability_is_one :
  basketball_game_probability = 1 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_is_one_l3646_364639


namespace NUMINAMATH_CALUDE_bin_drawing_probability_l3646_364625

def bin_probability : ℚ :=
  let total_balls : ℕ := 20
  let black_balls : ℕ := 10
  let white_balls : ℕ := 10
  let drawn_balls : ℕ := 4
  let favorable_outcomes : ℕ := (Nat.choose black_balls 2) * (Nat.choose white_balls 2)
  let total_outcomes : ℕ := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes

theorem bin_drawing_probability : bin_probability = 135 / 323 := by
  sorry

end NUMINAMATH_CALUDE_bin_drawing_probability_l3646_364625


namespace NUMINAMATH_CALUDE_trapezoid_semicircle_area_l3646_364626

/-- Represents a trapezoid with semicircles on each side -/
structure TrapezoidWithSemicircles where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the area of the region bounded by the semicircles -/
noncomputable def boundedArea (t : TrapezoidWithSemicircles) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_semicircle_area 
  (t : TrapezoidWithSemicircles) 
  (h1 : t.side1 = 10) 
  (h2 : t.side2 = 10) 
  (h3 : t.side3 = 10) 
  (h4 : t.side4 = 22) : 
  boundedArea t = 128 + 60.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_semicircle_area_l3646_364626


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l3646_364673

/-- Represents a right-angled triangle with side lengths a, b, and c (hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2

theorem right_triangle_side_lengths
  (t : RightTriangle)
  (leg_a : t.a = 10)
  (sum_squares : t.a^2 + t.b^2 + t.c^2 = 2050) :
  t.b = Real.sqrt 925 ∧ t.c = Real.sqrt 1025 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_side_lengths_l3646_364673


namespace NUMINAMATH_CALUDE_problem_statement_l3646_364640

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3646_364640


namespace NUMINAMATH_CALUDE_coffee_shop_spending_prove_coffee_shop_spending_l3646_364661

theorem coffee_shop_spending : ℝ → ℝ → Prop :=
  fun (ben_spent david_spent : ℝ) =>
    (david_spent = ben_spent / 2) →
    (ben_spent = david_spent + 15) →
    (ben_spent + david_spent = 45)

/-- Proof of the coffee shop spending theorem -/
theorem prove_coffee_shop_spending :
  ∃ (ben_spent david_spent : ℝ),
    coffee_shop_spending ben_spent david_spent :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_prove_coffee_shop_spending_l3646_364661


namespace NUMINAMATH_CALUDE_carly_bbq_cooking_time_l3646_364609

/-- Represents the cooking scenario for Carly's BBQ --/
structure BBQScenario where
  cook_time_per_side : ℕ
  burgers_per_batch : ℕ
  total_guests : ℕ
  guests_wanting_two : ℕ
  guests_wanting_one : ℕ

/-- Calculates the total cooking time for all burgers --/
def total_cooking_time (scenario : BBQScenario) : ℕ :=
  let total_burgers := 2 * scenario.guests_wanting_two + scenario.guests_wanting_one
  let num_batches := (total_burgers + scenario.burgers_per_batch - 1) / scenario.burgers_per_batch
  num_batches * (2 * scenario.cook_time_per_side)

/-- Theorem stating that the total cooking time for Carly's scenario is 72 minutes --/
theorem carly_bbq_cooking_time :
  total_cooking_time {
    cook_time_per_side := 4,
    burgers_per_batch := 5,
    total_guests := 30,
    guests_wanting_two := 15,
    guests_wanting_one := 15
  } = 72 := by
  sorry

end NUMINAMATH_CALUDE_carly_bbq_cooking_time_l3646_364609


namespace NUMINAMATH_CALUDE_cube_sum_ge_triple_product_l3646_364674

theorem cube_sum_ge_triple_product (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_triple_product_l3646_364674


namespace NUMINAMATH_CALUDE_sum_specific_terms_l3646_364669

/-- Given a sequence {a_n} where S_n = n^2 - 1 for n ∈ ℕ+, prove a_1 + a_3 + a_5 + a_7 + a_9 = 44 -/
theorem sum_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ+, S n = n^2 - 1) → 
  (∀ n : ℕ+, S n - S (n-1) = a n) → 
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 := by
sorry

end NUMINAMATH_CALUDE_sum_specific_terms_l3646_364669


namespace NUMINAMATH_CALUDE_vodka_mixture_profit_l3646_364605

/-- Represents the profit percentage of a mixture of two vodkas -/
def mixtureProfitPercentage (profit1 profit2 : ℚ) (increase1 increase2 : ℚ) : ℚ :=
  (profit1 * increase1 + profit2 * increase2) / 2

theorem vodka_mixture_profit :
  let initialProfit1 : ℚ := 10 / 100
  let initialProfit2 : ℚ := 40 / 100
  let increase1 : ℚ := 4 / 3
  let increase2 : ℚ := 5 / 3
  mixtureProfitPercentage initialProfit1 initialProfit2 increase1 increase2 = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_vodka_mixture_profit_l3646_364605


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l3646_364697

/-- Proves that the minimum bailing rate to reach shore without sinking is 10.75 gallons per minute --/
theorem minimum_bailing_rate 
  (distance : ℝ) 
  (intake_rate : ℝ) 
  (max_water : ℝ) 
  (rowing_speed : ℝ) 
  (h1 : distance = 2) 
  (h2 : intake_rate = 12) 
  (h3 : max_water = 50) 
  (h4 : rowing_speed = 3) : 
  ∃ (bailing_rate : ℝ), 
    bailing_rate ≥ 10.75 ∧ 
    bailing_rate < intake_rate ∧
    (distance / rowing_speed) * 60 * (intake_rate - bailing_rate) ≤ max_water :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l3646_364697


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3646_364645

theorem sum_of_a_and_b (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 - b^2 = -12) : a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3646_364645


namespace NUMINAMATH_CALUDE_number_difference_l3646_364602

/-- Given two positive integers where the larger number is 1596,
    and when divided by the smaller number results in a quotient of 6 and a remainder of 15,
    prove that the difference between these two numbers is equal to the calculated difference. -/
theorem number_difference (smaller larger : ℕ) (h1 : larger = 1596) 
    (h2 : larger = 6 * smaller + 15) : larger - smaller = larger - smaller := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3646_364602


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l3646_364607

/-- The function f(x) = sin(x/2) + cos(x/2) is monotonically increasing 
    on the intervals [4kπ - 3π/2, 4kπ + π/2] for all integer k. -/
theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x => Real.sin (x/2) + Real.cos (x/2))
    (Set.Icc (4 * k * Real.pi - 3 * Real.pi / 2) (4 * k * Real.pi + Real.pi / 2)) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l3646_364607


namespace NUMINAMATH_CALUDE_college_running_survey_l3646_364686

/-- Represents the sample data for running mileage --/
structure SampleData where
  male_0_30 : ℕ
  male_30_60 : ℕ
  male_60_90 : ℕ
  male_90_plus : ℕ
  female_0_30 : ℕ
  female_30_60 : ℕ
  female_60_90 : ℕ
  female_90_plus : ℕ

/-- Theorem representing the problem and its solution --/
theorem college_running_survey (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
    (sample : SampleData) :
    total_students = 1000 →
    male_students = 640 →
    female_students = 360 →
    sample.male_30_60 = 12 →
    sample.male_60_90 = 10 →
    sample.male_90_plus = 5 →
    sample.female_0_30 = 6 →
    sample.female_30_60 = 6 →
    sample.female_60_90 = 4 →
    sample.female_90_plus = 2 →
    (∃ (a : ℕ),
      sample.male_0_30 = a ∧
      a = 5 ∧
      ((a + 12 + 10 + 5 : ℚ) / (6 + 6 + 4 + 2) = 640 / 360) ∧
      (a * 1000 / (a + 12 + 10 + 5 + 6 + 6 + 4 + 2) = 100)) ∧
    (∃ (X : Fin 4 → ℚ),
      X 1 = 1/7 ∧ X 2 = 4/7 ∧ X 3 = 2/7 ∧
      (X 1 + X 2 + X 3 = 1) ∧
      (1 * X 1 + 2 * X 2 + 3 * X 3 = 15/7)) := by
  sorry


end NUMINAMATH_CALUDE_college_running_survey_l3646_364686


namespace NUMINAMATH_CALUDE_correlation_theorem_l3646_364641

-- Define the types for our quantities
def Time := ℝ
def Displacement := ℝ
def Grade := ℝ
def Weight := ℝ
def DrunkDrivers := ℕ
def TrafficAccidents := ℕ
def Volume := ℝ

-- Define a type for pairs of quantities
structure QuantityPair where
  first : Type
  second : Type

-- Define our pairs
def uniformMotionPair : QuantityPair := ⟨Time, Displacement⟩
def gradeWeightPair : QuantityPair := ⟨Grade, Weight⟩
def drunkDriverAccidentPair : QuantityPair := ⟨DrunkDrivers, TrafficAccidents⟩
def volumeWeightPair : QuantityPair := ⟨Volume, Weight⟩

-- Define a predicate for correlation
def hasCorrelation (pair : QuantityPair) : Prop := sorry

-- Theorem statement
theorem correlation_theorem :
  ¬ hasCorrelation uniformMotionPair ∧
  ¬ hasCorrelation gradeWeightPair ∧
  hasCorrelation drunkDriverAccidentPair ∧
  ¬ hasCorrelation volumeWeightPair :=
sorry

end NUMINAMATH_CALUDE_correlation_theorem_l3646_364641


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l3646_364684

theorem factorization_of_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l3646_364684


namespace NUMINAMATH_CALUDE_remainder_a_squared_minus_3b_l3646_364636

theorem remainder_a_squared_minus_3b (a b : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 5) 
  (h_ineq : a^2 > 3*b) : 
  (a^2 - 3*b) % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_a_squared_minus_3b_l3646_364636


namespace NUMINAMATH_CALUDE_largest_valid_p_l3646_364671

def is_valid_p (p : ℝ) : Prop :=
  p > 1 ∧ ∀ a b c : ℝ, 
    1/p ≤ a ∧ a ≤ p ∧
    1/p ≤ b ∧ b ≤ p ∧
    1/p ≤ c ∧ c ≤ p →
    9 * (a*b + b*c + c*a) * (a^2 + b^2 + c^2) ≥ (a + b + c)^4

theorem largest_valid_p :
  ∃ p : ℝ, p = Real.sqrt (4 + 3 * Real.sqrt 2) ∧
    is_valid_p p ∧
    ∀ q : ℝ, q > p → ¬is_valid_p q :=
sorry

end NUMINAMATH_CALUDE_largest_valid_p_l3646_364671


namespace NUMINAMATH_CALUDE_equation_solution_l3646_364657

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 :=
by
  use -13/4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3646_364657


namespace NUMINAMATH_CALUDE_volume_equals_target_l3646_364638

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of points inside or within one unit of a parallelepiped -/
def volume_with_buffer (p : Parallelepiped) : ℝ := sorry

/-- The specific parallelepiped in the problem -/
def problem_parallelepiped : Parallelepiped :=
  { length := 2,
    width := 3,
    height := 4 }

theorem volume_equals_target : 
  volume_with_buffer problem_parallelepiped = (456 + 31 * Real.pi) / 6 := by sorry

end NUMINAMATH_CALUDE_volume_equals_target_l3646_364638


namespace NUMINAMATH_CALUDE_lottery_probabilities_l3646_364646

def total_numbers : ℕ := 10
def numbers_per_ticket : ℕ := 5
def numbers_drawn : ℕ := 4

def probability_four_match : ℚ := 1 / 21
def probability_two_match : ℚ := 10 / 21

theorem lottery_probabilities :
  (total_numbers = 10) →
  (numbers_per_ticket = 5) →
  (numbers_drawn = 4) →
  (probability_four_match = 1 / 21) ∧
  (probability_two_match = 10 / 21) := by
  sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l3646_364646


namespace NUMINAMATH_CALUDE_price_reduction_effect_l3646_364653

theorem price_reduction_effect (P S : ℝ) (P_reduced : ℝ) (S_increased : ℝ) :
  P_reduced = 0.8 * P →
  S_increased = 1.8 * S →
  P_reduced * S_increased = 1.44 * P * S :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l3646_364653


namespace NUMINAMATH_CALUDE_modulus_of_z_l3646_364699

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 + I) * (1 - z) = 1) : abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3646_364699


namespace NUMINAMATH_CALUDE_total_cows_l3646_364615

theorem total_cows (cows_per_herd : ℕ) (num_herds : ℕ) (h1 : cows_per_herd = 40) (h2 : num_herds = 8) :
  cows_per_herd * num_herds = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_l3646_364615


namespace NUMINAMATH_CALUDE_ab_value_l3646_364659

theorem ab_value (a b : ℝ) (h : b = Real.sqrt (1 - 2*a) + Real.sqrt (2*a - 1) + 3) : 
  a^b = (1/8 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ab_value_l3646_364659


namespace NUMINAMATH_CALUDE_a_income_l3646_364633

def income_ratio : ℚ := 5 / 4
def expenditure_ratio : ℚ := 3 / 2
def savings : ℕ := 1600

theorem a_income (a_income b_income a_expenditure b_expenditure : ℚ) 
  (h1 : a_income / b_income = income_ratio)
  (h2 : a_expenditure / b_expenditure = expenditure_ratio)
  (h3 : a_income - a_expenditure = savings)
  (h4 : b_income - b_expenditure = savings) :
  a_income = 4000 := by
  sorry

end NUMINAMATH_CALUDE_a_income_l3646_364633


namespace NUMINAMATH_CALUDE_expression_simplification_l3646_364687

theorem expression_simplification (m : ℝ) 
  (h1 : (m + 2) * (m - 3) = 0) 
  (h2 : m ≠ 3) : 
  ((m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m - 3)) / (m^2 / m^3) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3646_364687


namespace NUMINAMATH_CALUDE_line_intersecting_circle_slope_l3646_364647

/-- A line passing through (4,0) intersecting the circle (x-2)^2 + y^2 = 1 has slope -√3/3 or √3/3 -/
theorem line_intersecting_circle_slope :
  ∀ (k : ℝ), 
    (∃ (x y : ℝ), y = k * (x - 4) ∧ (x - 2)^2 + y^2 = 1) →
    (k = -Real.sqrt 3 / 3 ∨ k = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_intersecting_circle_slope_l3646_364647


namespace NUMINAMATH_CALUDE_ball_count_l3646_364693

theorem ball_count (white blue red : ℕ) : 
  blue = white + 12 →
  red = 2 * blue →
  white = 16 →
  white + blue + red = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_count_l3646_364693


namespace NUMINAMATH_CALUDE_shifted_proportional_function_l3646_364681

/-- Given a proportional function y = -2x that is shifted up by 3 units,
    the resulting function is y = -2x + 3. -/
theorem shifted_proportional_function :
  let f : ℝ → ℝ := λ x ↦ -2 * x
  let shift : ℝ := 3
  let g : ℝ → ℝ := λ x ↦ f x + shift
  ∀ x : ℝ, g x = -2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_shifted_proportional_function_l3646_364681


namespace NUMINAMATH_CALUDE_find_x_l3646_364690

theorem find_x : ∃ x : ℚ, (3 * x + 4) / 6 = 15 ∧ x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3646_364690


namespace NUMINAMATH_CALUDE_exists_a_for_even_f_l3646_364656

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem exists_a_for_even_f : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_for_even_f_l3646_364656


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l3646_364688

/-- The area of a rhombus formed by the intersection of two equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_side : ℝ := square_side
  let triangle_height : ℝ := (Real.sqrt 3 / 2) * triangle_side
  let rhombus_diagonal1 : ℝ := square_side
  let rhombus_diagonal2 : ℝ := triangle_height
  let rhombus_area : ℝ := (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2
  rhombus_area = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l3646_364688


namespace NUMINAMATH_CALUDE_largest_number_in_set_l3646_364680

/-- Given a = -3, -4a is the largest number in the set {-4a, 3a, 36/a, a^3, 2} -/
theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  (-4 * a) = max (-4 * a) (max (3 * a) (max (36 / a) (max (a ^ 3) 2))) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l3646_364680


namespace NUMINAMATH_CALUDE_right_triangle_legs_l3646_364651

theorem right_triangle_legs (c n : ℝ) (h1 : c > 0) (h2 : n > 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    (n / Real.sqrt 3)^2 = a * b * (1 - ((a + b) / c)^2) ∧
    a = n / 2 ∧ b = c * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l3646_364651


namespace NUMINAMATH_CALUDE_count_monomials_l3646_364621

/-- A function that determines if an algebraic expression is a monomial -/
def isMonomial (expr : String) : Bool :=
  match expr with
  | "(m+n)/2" => false
  | "2x^2y" => true
  | "1/x" => false
  | "-5" => true
  | "a" => true
  | _ => false

/-- The set of given algebraic expressions -/
def expressions : List String := ["(m+n)/2", "2x^2y", "1/x", "-5", "a"]

/-- Theorem stating that the number of monomials in the given set of expressions is 3 -/
theorem count_monomials :
  (expressions.filter isMonomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_monomials_l3646_364621


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3646_364695

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3646_364695


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l3646_364610

/-- The carpet area required for a rectangular room -/
def carpet_area (length width : ℝ) (wastage_factor : ℝ) : ℝ :=
  length * width * (1 + wastage_factor)

/-- Theorem: The carpet area for a 15 ft by 9 ft room with 10% wastage is 148.5 sq ft -/
theorem carpet_area_calculation :
  carpet_area 15 9 0.1 = 148.5 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_calculation_l3646_364610


namespace NUMINAMATH_CALUDE_angle_range_l3646_364600

theorem angle_range (α : Real) (h1 : α ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : Real.sin α < 0) (h3 : Real.cos α > 0) :
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_angle_range_l3646_364600


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3646_364664

def f (x : ℝ) := x^3 - 12*x

theorem max_min_f_on_interval :
  let a := -3
  let b := 5
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 65 ∧ min = -16 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3646_364664


namespace NUMINAMATH_CALUDE_composite_form_l3646_364655

theorem composite_form (x : ℤ) (m n : ℕ) (hm : m > 0) (hn : n ≥ 0) :
  x^(4*m) + 2^(4*n + 2) = (x^(2*m) + 2^(2*n + 1) + 2^(n + 1) * x^m) * ((x^m - 2^n)^2 + 2^(2*n)) :=
by sorry

end NUMINAMATH_CALUDE_composite_form_l3646_364655


namespace NUMINAMATH_CALUDE_jude_bottle_cap_trading_l3646_364679

/-- Jude's bottle cap trading problem -/
theorem jude_bottle_cap_trading
  (initial_caps : ℕ)
  (car_cost : ℕ)
  (truck_cost : ℕ)
  (trucks_bought : ℕ)
  (total_vehicles : ℕ)
  (h1 : initial_caps = 100)
  (h2 : car_cost = 5)
  (h3 : truck_cost = 6)
  (h4 : trucks_bought = 10)
  (h5 : total_vehicles = 16) :
  (car_cost * (total_vehicles - trucks_bought) : ℚ) / (initial_caps - truck_cost * trucks_bought) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_jude_bottle_cap_trading_l3646_364679


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3646_364617

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧ 
  (∀ (x : ℝ), x ≠ 0 → f (x + 1/x^2) = f x + (f (1/x))^2) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3646_364617


namespace NUMINAMATH_CALUDE_profit_range_max_avg_profit_l3646_364603

/-- Cumulative profit function -/
def profit (x : ℕ) : ℚ :=
  -1/2 * x^2 + 60*x - 800

/-- Average daily profit function -/
def avgProfit (x : ℕ) : ℚ :=
  profit x / x

theorem profit_range (x : ℕ) (hx : x > 0) :
  profit x > 800 ↔ x > 40 ∧ x < 80 :=
sorry

theorem max_avg_profit :
  ∃ (x : ℕ), x > 0 ∧ ∀ (y : ℕ), y > 0 → avgProfit x ≥ avgProfit y ∧ x = 400 :=
sorry

end NUMINAMATH_CALUDE_profit_range_max_avg_profit_l3646_364603


namespace NUMINAMATH_CALUDE_hundred_chicken_equations_l3646_364675

def hundred_chicken_problem (x y : ℝ) : Prop :=
  (x + y + 81 = 100) ∧ (5*x + 3*y + (1/3) * 81 = 100)

theorem hundred_chicken_equations :
  ∀ x y : ℝ,
  (x ≥ 0) → (y ≥ 0) →
  (x + y + 81 = 100) →
  (5*x + 3*y + 27 = 100) →
  hundred_chicken_problem x y :=
by
  sorry

end NUMINAMATH_CALUDE_hundred_chicken_equations_l3646_364675


namespace NUMINAMATH_CALUDE_proportional_function_k_value_l3646_364620

/-- A proportional function passing through a specific point -/
def proportional_function (k : ℝ) (x : ℝ) : ℝ := k * x

theorem proportional_function_k_value :
  ∀ k : ℝ,
  k ≠ 0 →
  proportional_function k 3 = -6 →
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_proportional_function_k_value_l3646_364620


namespace NUMINAMATH_CALUDE_mother_age_proof_l3646_364623

def id_number : ℕ := 6101131197410232923
def current_year : ℕ := 2014

def extract_birth_year (id : ℕ) : ℕ :=
  (id / 10^13) % 10000

def calculate_age (birth_year current_year : ℕ) : ℕ :=
  current_year - birth_year

theorem mother_age_proof :
  calculate_age (extract_birth_year id_number) current_year = 40 := by
  sorry

end NUMINAMATH_CALUDE_mother_age_proof_l3646_364623


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3646_364654

-- Problem 1
theorem problem_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (-2 * a^2)^2 * (-b^2) / (4 * a^3 * b^2) = -a := by sorry

-- Problem 2
theorem problem_2 : 2023^2 - 2021 * 2025 = 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3646_364654


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l3646_364658

/-- Given a function g: ℝ → ℝ, prove that if (2,5) lies on the graph of y = g(x),
    then (1,8) lies on the graph of 4y = 5g(3x-1) + 7, and the sum of the coordinates of (1,8) is 9. -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 2 = 5) :
  4 * 8 = 5 * g (3 * 1 - 1) + 7 ∧ 1 + 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l3646_364658


namespace NUMINAMATH_CALUDE_orchestra_members_count_l3646_364627

theorem orchestra_members_count :
  ∃! n : ℕ, 150 < n ∧ n < 250 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 7 = 4 ∧
    n = 158 := by
  sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l3646_364627


namespace NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l3646_364608

theorem max_value_sum_of_square_roots (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
  (h_sum : a + b + c = 7) : 
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 6 ∧
  (∃ (a₀ b₀ c₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 7 ∧
    Real.sqrt (3 * a₀ + 1) + Real.sqrt (3 * b₀ + 1) + Real.sqrt (3 * c₀ + 1) = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l3646_364608
