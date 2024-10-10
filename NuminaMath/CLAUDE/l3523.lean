import Mathlib

namespace digit_150_of_17_150_l3523_352339

/-- The decimal representation of 17/150 -/
def decimal_rep : ℚ := 17 / 150

/-- The nth digit after the decimal point in a rational number -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_150_of_17_150 :
  nth_digit decimal_rep 150 = 3 :=
sorry

end digit_150_of_17_150_l3523_352339


namespace max_stamps_purchased_l3523_352397

/-- Given a stamp price of 45 cents and $50 to spend, 
    the maximum number of stamps that can be purchased is 111. -/
theorem max_stamps_purchased (stamp_price : ℕ) (budget : ℕ) : 
  stamp_price = 45 → budget = 5000 → 
  (∀ n : ℕ, n * stamp_price ≤ budget → n ≤ 111) ∧ 
  111 * stamp_price ≤ budget :=
by sorry

end max_stamps_purchased_l3523_352397


namespace father_age_l3523_352387

/-- Represents the ages and relationships of family members -/
structure FamilyAges where
  peter : ℕ
  jane : ℕ
  harriet : ℕ
  emily : ℕ
  mother : ℕ
  aunt_lucy : ℕ
  father : ℕ

/-- The conditions given in the problem -/
def family_conditions (f : FamilyAges) : Prop :=
  f.peter + 12 = 2 * (f.harriet + 12) ∧
  f.jane = f.emily + 10 ∧
  3 * f.peter = f.mother ∧
  f.mother = 60 ∧
  f.peter = f.jane + 5 ∧
  f.aunt_lucy = 52 ∧
  f.aunt_lucy = f.mother + 4 ∧
  f.father = f.aunt_lucy + 20

/-- The theorem to be proved -/
theorem father_age (f : FamilyAges) : 
  family_conditions f → f.father = 72 := by
  sorry

end father_age_l3523_352387


namespace exactly_one_topic_not_chosen_l3523_352371

/-- The number of ways for n teachers to choose from m topics with replacement. -/
def choose_with_replacement (n m : ℕ) : ℕ := m ^ n

/-- The number of ways to arrange n items. -/
def arrangement (n : ℕ) : ℕ := n.factorial

/-- The number of ways for n teachers to choose from m topics with replacement,
    such that exactly one topic is not chosen. -/
def one_topic_not_chosen (n m : ℕ) : ℕ :=
  choose_with_replacement n m -
  (m * choose_with_replacement (n - 1) (m - 1)) -
  arrangement m

theorem exactly_one_topic_not_chosen :
  one_topic_not_chosen 4 4 = 112 := by sorry

end exactly_one_topic_not_chosen_l3523_352371


namespace product_without_x_terms_l3523_352362

theorem product_without_x_terms (m n : ℝ) : 
  (∀ x : ℝ, (x + 2*m) * (x^2 - x + 1/2*n) = x^3 + 2*m*n) → 
  m^2023 * n^2022 = 1/2 := by
sorry

end product_without_x_terms_l3523_352362


namespace constant_sequence_l3523_352395

theorem constant_sequence (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i > j → ((i - j)^(2*(i - j)) + 1) ∣ (a i - a j)) :
  ∀ n : ℕ, n ≥ 1 → a n = a 1 :=
by sorry

end constant_sequence_l3523_352395


namespace complex_expression_equals_100_algebraic_expression_simplification_l3523_352358

-- Problem 1
theorem complex_expression_equals_100 :
  (2 * (7 / 9 : ℝ)) ^ (1 / 2 : ℝ) + (1 / 10 : ℝ) ^ (-2 : ℝ) + 
  (2 * (10 / 27 : ℝ)) ^ (-(2 / 3) : ℝ) - 3 * (Real.pi ^ (0 : ℝ)) + 
  (37 / 48 : ℝ) = 100 := by sorry

-- Problem 2
theorem algebraic_expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (a ^ (2 / 3)) * (b ^ (1 / 2))) * (-6 * (a ^ (1 / 2)) * (b ^ (1 / 3))) / 
  (-3 * (a ^ (1 / 6)) * (b ^ (5 / 6))) = 4 * a := by sorry

end complex_expression_equals_100_algebraic_expression_simplification_l3523_352358


namespace simplify_polynomial_l3523_352367

theorem simplify_polynomial (x : ℝ) :
  2 * x * (5 * x^2 - 3 * x + 1) + 4 * (x^2 - 3 * x + 6) =
  10 * x^3 - 2 * x^2 - 10 * x + 24 := by
  sorry

end simplify_polynomial_l3523_352367


namespace photo_album_distribution_l3523_352306

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

end photo_album_distribution_l3523_352306


namespace divisibility_by_66_l3523_352361

theorem divisibility_by_66 : ∃ k : ℤ, 43^23 + 23^43 = 66 * k := by
  sorry

end divisibility_by_66_l3523_352361


namespace rectangle_max_area_rectangle_max_area_value_l3523_352323

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

end rectangle_max_area_rectangle_max_area_value_l3523_352323


namespace average_weight_increase_l3523_352346

theorem average_weight_increase (original_count : ℕ) (original_weight replaced_weight new_weight : ℝ) :
  original_count = 9 →
  replaced_weight = 65 →
  new_weight = 87.5 →
  (new_weight - replaced_weight) / original_count = 2.5 := by
sorry

end average_weight_increase_l3523_352346


namespace michaels_brother_money_l3523_352365

/-- Given that Michael has $42 and his brother has $17, Michael gives half his money to his brother,
    and his brother then buys $3 worth of candy, prove that his brother ends up with $35. -/
theorem michaels_brother_money (michael_initial : ℕ) (brother_initial : ℕ) 
    (candy_cost : ℕ) (h1 : michael_initial = 42) (h2 : brother_initial = 17) 
    (h3 : candy_cost = 3) : 
    brother_initial + michael_initial / 2 - candy_cost = 35 := by
  sorry

end michaels_brother_money_l3523_352365


namespace quadratic_equation_roots_condition_l3523_352324

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

end quadratic_equation_roots_condition_l3523_352324


namespace runners_meet_at_start_l3523_352390

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race -/
structure RaceState where
  runner_a : Runner
  runner_b : Runner
  time : ℝ

def track_length : ℝ := 300

/-- Function to update the race state after each meeting -/
def update_race_state (state : RaceState) : RaceState :=
  sorry

/-- Function to check if both runners are at the starting point -/
def at_start (state : RaceState) : Bool :=
  sorry

/-- Theorem stating that the runners meet at the starting point after 250 seconds -/
theorem runners_meet_at_start :
  let initial_state : RaceState := {
    runner_a := { speed := 2, direction := true },
    runner_b := { speed := 4, direction := false },
    time := 0
  }
  let final_state := update_race_state initial_state
  (at_start final_state ∧ final_state.time = 250) := by
  sorry

end runners_meet_at_start_l3523_352390


namespace pizza_party_children_count_l3523_352332

theorem pizza_party_children_count (total : ℕ) (children : ℕ) (adults : ℕ) : 
  total = 120 →
  children = 2 * adults →
  total = children + adults →
  children = 80 := by
sorry

end pizza_party_children_count_l3523_352332


namespace someone_next_to_two_economists_l3523_352375

/-- Represents the profession of a person -/
inductive Profession
| Accountant
| Manager
| Economist

/-- Represents a circular arrangement of people -/
def CircularArrangement := List Profession

/-- Counts the number of accountants sitting next to at least one economist -/
def accountantsNextToEconomist (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of managers sitting next to at least one economist -/
def managersNextToEconomist (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if there's someone sitting next to two economists -/
def someoneNextToTwoEconomists (arrangement : CircularArrangement) : Bool :=
  sorry

theorem someone_next_to_two_economists 
  (arrangement : CircularArrangement) : 
  accountantsNextToEconomist arrangement = 20 →
  managersNextToEconomist arrangement = 25 →
  someoneNextToTwoEconomists arrangement = true :=
by sorry

end someone_next_to_two_economists_l3523_352375


namespace matt_received_more_than_lauren_l3523_352335

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

end matt_received_more_than_lauren_l3523_352335


namespace complex_modulus_equation_l3523_352341

theorem complex_modulus_equation (a : ℝ) : 
  Complex.abs ((5 : ℂ) / (2 + Complex.I) + a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_modulus_equation_l3523_352341


namespace quadratic_equations_solutions_l3523_352326

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 3 = 0) ∧ 
  (∃ x : ℝ, x*(x-2) = 2*(2-x)) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1)) ∧
  (∀ x : ℝ, x*(x-2) = 2*(2-x) ↔ (x = 2 ∨ x = -2)) :=
by sorry

end quadratic_equations_solutions_l3523_352326


namespace ralph_cards_l3523_352382

/-- The number of cards Ralph has after various changes. -/
def final_cards (initial : ℕ) (from_father : ℕ) (from_sister : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_father + from_sister - traded - lost

/-- Theorem stating that Ralph ends up with 12 cards given the specific card changes. -/
theorem ralph_cards : final_cards 4 8 5 3 2 = 12 := by
  sorry

end ralph_cards_l3523_352382


namespace farm_animals_l3523_352381

/-- Given a farm with hens and cows, prove the number of hens -/
theorem farm_animals (total_heads : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) :
  total_heads = 44 →
  total_feet = 140 →
  total_heads = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 18 := by
  sorry

end farm_animals_l3523_352381


namespace line_shift_l3523_352378

/-- The vertical shift of a line -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f x + shift

/-- The original line equation -/
def original_line : ℝ → ℝ := fun x ↦ 3 * x - 2

/-- Theorem: Moving the line y = 3x - 2 up by 6 units results in y = 3x + 4 -/
theorem line_shift :
  vertical_shift original_line 6 = fun x ↦ 3 * x + 4 := by
  sorry

end line_shift_l3523_352378


namespace paper_clips_in_2_cases_l3523_352393

/-- The number of paper clips in 2 cases -/
def paperClipsIn2Cases (c b : ℕ) : ℕ := 2 * c * b * 400

/-- Theorem: The number of paper clips in 2 cases is 2 * c * b * 400 -/
theorem paper_clips_in_2_cases (c b : ℕ) : paperClipsIn2Cases c b = 2 * c * b * 400 := by
  sorry

end paper_clips_in_2_cases_l3523_352393


namespace min_cost_2009_l3523_352309

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


end min_cost_2009_l3523_352309


namespace perpendicular_lines_l3523_352313

/-- Two lines y = ax - 2 and y = x + 1 are perpendicular if and only if a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∧ y = x + 1) → a = -1 := by
  sorry

end perpendicular_lines_l3523_352313


namespace no_equal_volume_increase_l3523_352317

theorem no_equal_volume_increase (x : ℝ) : ¬ (
  let R : ℝ := 10
  let H : ℝ := 5
  let V (r h : ℝ) := Real.pi * r^2 * h
  V (R + x) H - V R H = V R (H + x) - V R H
) := by sorry

end no_equal_volume_increase_l3523_352317


namespace trig_expression_range_l3523_352366

theorem trig_expression_range (C : ℝ) (h : 0 < C ∧ C < π) :
  ∃ (lower upper : ℝ), lower = -1 ∧ upper = Real.sqrt 2 ∧
  -1 < (2 * Real.cos (2 * C) / Real.tan C) + 1 ∧
  (2 * Real.cos (2 * C) / Real.tan C) + 1 ≤ Real.sqrt 2 := by
sorry

end trig_expression_range_l3523_352366


namespace factor_implies_b_value_l3523_352334

theorem factor_implies_b_value (b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, 9*x^2 + b*x + 44 = (3*x + 4) * k) → b = 45 := by
  sorry

end factor_implies_b_value_l3523_352334


namespace cups_per_girl_l3523_352331

theorem cups_per_girl (total_students : Nat) (boys : Nat) (cups_per_boy : Nat) (total_cups : Nat)
  (h1 : total_students = 30)
  (h2 : boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_cups = 90)
  (h5 : boys * 2 = total_students - boys) :
  (total_cups - boys * cups_per_boy) / (total_students - boys) = 2 := by
  sorry

end cups_per_girl_l3523_352331


namespace marla_nightly_cost_l3523_352303

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

end marla_nightly_cost_l3523_352303


namespace average_distance_is_600_l3523_352342

/-- The length of one lap around the block in meters -/
def block_length : ℕ := 200

/-- The number of times Johnny runs around the block -/
def johnny_laps : ℕ := 4

/-- The number of times Mickey runs around the block -/
def mickey_laps : ℕ := johnny_laps / 2

/-- The total distance run by Johnny in meters -/
def johnny_distance : ℕ := johnny_laps * block_length

/-- The total distance run by Mickey in meters -/
def mickey_distance : ℕ := mickey_laps * block_length

/-- The average distance run by Johnny and Mickey in meters -/
def average_distance : ℕ := (johnny_distance + mickey_distance) / 2

theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end average_distance_is_600_l3523_352342


namespace inequality_equivalence_l3523_352391

theorem inequality_equivalence (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  (x * z^2 / z > y * z^2 / z) ↔ (x > y) := by
  sorry

end inequality_equivalence_l3523_352391


namespace solution_value_l3523_352396

theorem solution_value (a : ℝ) : (2 * (-1) + 3 * a = 4) → a = 2 := by
  sorry

end solution_value_l3523_352396


namespace right_triangle_division_area_ratio_l3523_352304

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

end right_triangle_division_area_ratio_l3523_352304


namespace min_value_theorem_l3523_352349

theorem min_value_theorem (m n p x y z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_mnp : m * n * p = 8) (h_xyz : x * y * z = 8) :
  let f := x^2 + y^2 + z^2 + m*x*y + n*x*z + p*y*z
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → f ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') ∧
  (m = 2 ∧ n = 2 ∧ p = 2 → ∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → 
    36 ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') ∧
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x' * y' * z' = 8 → 
    6 * (2^(1/3 : ℝ)) * (m^(2/3 : ℝ) + n^(2/3 : ℝ) + p^(2/3 : ℝ)) ≤ x'^2 + y'^2 + z'^2 + m*x'*y' + n*x'*z' + p*y'*z') :=
by
  sorry

end min_value_theorem_l3523_352349


namespace cyclic_matrix_squared_identity_l3523_352373

/-- A 4x4 complex matrix with a cyclic structure -/
def CyclicMatrix (a b c d : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  !![a, b, c, d;
     b, c, d, a;
     c, d, a, b;
     d, a, b, c]

theorem cyclic_matrix_squared_identity
  (a b c d : ℂ)
  (h1 : (CyclicMatrix a b c d) ^ 2 = 1)
  (h2 : a * b * c * d = 1) :
  a ^ 4 + b ^ 4 + c ^ 4 + d ^ 4 = 2 := by
  sorry

end cyclic_matrix_squared_identity_l3523_352373


namespace line_is_intersection_l3523_352384

/-- The line of intersection of two planes -/
def line_of_intersection (p₁ p₂ : ℝ → ℝ → ℝ → Prop) : ℝ → ℝ → ℝ → Prop :=
  λ x y z => (x + 3) / (-3) = y / (-4) ∧ y / (-4) = z / (-9)

/-- First plane equation -/
def plane1 : ℝ → ℝ → ℝ → Prop :=
  λ x y z => 2*x + 3*y - 2*z + 6 = 0

/-- Second plane equation -/
def plane2 : ℝ → ℝ → ℝ → Prop :=
  λ x y z => x - 3*y + z + 3 = 0

/-- Theorem stating that the line is the intersection of the two planes -/
theorem line_is_intersection :
  ∀ x y z, line_of_intersection plane1 plane2 x y z ↔ (plane1 x y z ∧ plane2 x y z) :=
sorry

end line_is_intersection_l3523_352384


namespace arithmetic_and_geometric_sequences_l3523_352351

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

end arithmetic_and_geometric_sequences_l3523_352351


namespace bouncing_ball_distance_l3523_352355

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The bouncing ball theorem -/
theorem bouncing_ball_distance :
  let initialHeight : ℝ := 120
  let reboundFactor : ℝ := 0.75
  let bounces : ℕ := 5
  totalDistance initialHeight reboundFactor bounces = 612.1875 := by
  sorry

end bouncing_ball_distance_l3523_352355


namespace functional_equation_solution_l3523_352345

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = 1) :=
by sorry

end functional_equation_solution_l3523_352345


namespace product_selection_events_l3523_352374

structure ProductSelection where
  total : Nat
  genuine : Nat
  defective : Nat
  selected : Nat

def is_random_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∃ (outcome : Nat), event outcome ∧
  ∃ (outcome : Nat), ¬event outcome

def is_impossible_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∀ (outcome : Nat), ¬event outcome

def is_certain_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∀ (outcome : Nat), event outcome

def all_genuine (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome = ps.genuine.choose ps.selected

def at_least_one_defective (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome > 0

def all_defective (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome = ps.defective.choose ps.selected

def at_least_one_genuine (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome < ps.selected

theorem product_selection_events (ps : ProductSelection) 
  (h1 : ps.total = 12)
  (h2 : ps.genuine = 10)
  (h3 : ps.defective = 2)
  (h4 : ps.selected = 3)
  (h5 : ps.total = ps.genuine + ps.defective) :
  is_random_event ps (all_genuine ps) ∧
  is_random_event ps (at_least_one_defective ps) ∧
  is_impossible_event ps (all_defective ps) ∧
  is_certain_event ps (at_least_one_genuine ps) := by
  sorry

end product_selection_events_l3523_352374


namespace sequence_property_main_theorem_l3523_352316

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

end sequence_property_main_theorem_l3523_352316


namespace constant_term_product_l3523_352357

variables (p q r : ℝ[X])

theorem constant_term_product (hp : p.coeff 0 = 5) (hr : r.coeff 0 = -15) (h_prod : r = p * q) :
  q.coeff 0 = -3 := by
  sorry

end constant_term_product_l3523_352357


namespace probability_no_twos_l3523_352310

def valid_id (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 5000 ∧ ¬(String.contains (toString n) '2')

def count_valid_ids : Nat :=
  (List.range 5000).filter valid_id |>.length

theorem probability_no_twos :
  count_valid_ids = 2916 →
  (count_valid_ids : ℚ) / 5000 = 729 / 1250 := by
  sorry

end probability_no_twos_l3523_352310


namespace hendecagon_diagonals_l3523_352302

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hendecagon is an 11-sided polygon -/
def hendecagon_sides : ℕ := 11

/-- The number of diagonals in a hendecagon is 44 -/
theorem hendecagon_diagonals : num_diagonals hendecagon_sides = 44 := by
  sorry

end hendecagon_diagonals_l3523_352302


namespace half_radius_circle_y_l3523_352308

-- Define the circles
def circle_x : Real → Prop := λ r => r > 0
def circle_y : Real → Prop := λ r => r > 0

-- Define the theorem
theorem half_radius_circle_y 
  (h_area : ∀ (rx ry : Real), circle_x rx → circle_y ry → π * rx^2 = π * ry^2)
  (h_circum : ∀ (rx : Real), circle_x rx → 2 * π * rx = 10 * π) :
  ∃ (ry : Real), circle_y ry ∧ ry / 2 = 2.5 := by
  sorry

end half_radius_circle_y_l3523_352308


namespace gold_bar_weight_l3523_352380

/-- Proves that in an arithmetic sequence of 5 terms where the first term is 4 
    and the last term is 2, the second term is 7/2. -/
theorem gold_bar_weight (a : Fin 5 → ℚ) 
  (h_arith : ∀ i j : Fin 5, a j - a i = (j - i : ℚ) * (a 1 - a 0))
  (h_first : a 0 = 4)
  (h_last : a 4 = 2) : 
  a 1 = 7/2 := by
  sorry

end gold_bar_weight_l3523_352380


namespace parallelogram_area_specific_vectors_l3523_352385

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogram_area (v w : Fin 2 → ℝ) : ℝ :=
  |v 0 * w 1 - v 1 * w 0|

theorem parallelogram_area_specific_vectors :
  let v : Fin 2 → ℝ := ![7, -4]
  let w : Fin 2 → ℝ := ![12, -1]
  parallelogram_area v w = 41 := by
  sorry

end parallelogram_area_specific_vectors_l3523_352385


namespace amoeba_growth_after_week_l3523_352344

def amoeba_population (initial_population : ℕ) (days : ℕ) : ℕ :=
  if days = 0 then
    initial_population
  else if days % 2 = 1 then
    2 * amoeba_population initial_population (days - 1)
  else
    3 * 2 * amoeba_population initial_population (days - 1)

theorem amoeba_growth_after_week :
  amoeba_population 4 7 = 13824 := by
  sorry

end amoeba_growth_after_week_l3523_352344


namespace relationship_theorem_l3523_352386

theorem relationship_theorem (x y z w : ℝ) :
  (x + y) / (y + z) = (z + w) / (w + x) →
  x = z ∨ x + y + w + z = 0 :=
by sorry

end relationship_theorem_l3523_352386


namespace range_of_a_l3523_352364

def A := {x : ℝ | -1 < x ∧ x < 6}
def B (a : ℝ) := {x : ℝ | x^2 - 2*x + 1 - a^2 ≥ 0}

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∉ A → x ∈ B a) ∧ (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → 
  0 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l3523_352364


namespace class_grade_point_average_l3523_352325

/-- Calculate the grade point average of a class given the distribution of grades --/
theorem class_grade_point_average 
  (total_students : ℕ) 
  (gpa_60_percent : ℚ) 
  (gpa_65_percent : ℚ) 
  (gpa_70_percent : ℚ) 
  (gpa_80_percent : ℚ) 
  (h1 : total_students = 120)
  (h2 : gpa_60_percent = 25 / 100)
  (h3 : gpa_65_percent = 35 / 100)
  (h4 : gpa_70_percent = 15 / 100)
  (h5 : gpa_80_percent = 1 - (gpa_60_percent + gpa_65_percent + gpa_70_percent))
  (h6 : gpa_60_percent + gpa_65_percent + gpa_70_percent + gpa_80_percent = 1) :
  let weighted_average := 
    (gpa_60_percent * 60 + gpa_65_percent * 65 + gpa_70_percent * 70 + gpa_80_percent * 80)
  weighted_average = 68.25 := by
  sorry

end class_grade_point_average_l3523_352325


namespace marbles_exceed_200_l3523_352318

theorem marbles_exceed_200 : ∃ k : ℕ, (∀ j : ℕ, j < k → 5 * 2^j ≤ 200) ∧ 5 * 2^k > 200 ∧ k = 6 := by
  sorry

end marbles_exceed_200_l3523_352318


namespace fraction_enlargement_l3523_352376

theorem fraction_enlargement (x y : ℝ) (h : 3 * x - y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - (3 * y)) = 3 * ((2 * x * y) / (3 * x - y)) :=
by sorry

end fraction_enlargement_l3523_352376


namespace quadratic_function_k_value_l3523_352350

theorem quadratic_function_k_value (a b c : ℤ) (k : ℤ) : 
  let f : ℝ → ℝ := λ x => (a * x^2 + b * x + c : ℝ)
  (f 1 = 0) →
  (60 < f 9 ∧ f 9 < 70) →
  (90 < f 10 ∧ f 10 < 100) →
  (10000 * k < f 100 ∧ f 100 < 10000 * (k + 1)) →
  k = 2 := by
sorry

end quadratic_function_k_value_l3523_352350


namespace temperatures_median_and_range_l3523_352392

def temperatures : List ℝ := [12, 9, 10, 6, 11, 12, 17]

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem temperatures_median_and_range :
  median temperatures = 11 ∧ range temperatures = 11 := by
  sorry

end temperatures_median_and_range_l3523_352392


namespace mixed_groups_count_l3523_352363

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

end mixed_groups_count_l3523_352363


namespace mothers_age_l3523_352369

/-- Given a person and their mother, with the following conditions:
  1. The person's present age is two-fifths of the age of his mother.
  2. After 10 years, the person will be one-half of the age of his mother.
  This theorem proves that the mother's present age is 50 years. -/
theorem mothers_age (person_age mother_age : ℕ) 
  (h1 : person_age = (2 * mother_age) / 5)
  (h2 : person_age + 10 = (mother_age + 10) / 2) : 
  mother_age = 50 := by
  sorry

end mothers_age_l3523_352369


namespace minimum_parents_needed_tour_parents_theorem_l3523_352398

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) : ℕ :=
  let total_people := num_children
  let cars_needed := (total_people + car_capacity - 1) / car_capacity
  cars_needed

theorem tour_parents_theorem :
  minimum_parents_needed 50 6 = 10 := by
  sorry

end minimum_parents_needed_tour_parents_theorem_l3523_352398


namespace lindas_tv_cost_l3523_352399

/-- The cost of Linda's TV purchase, given her original savings and furniture expenses -/
theorem lindas_tv_cost (original_savings : ℝ) (furniture_fraction : ℝ) : 
  original_savings = 800 →
  furniture_fraction = 3/4 →
  original_savings * (1 - furniture_fraction) = 200 := by
sorry

end lindas_tv_cost_l3523_352399


namespace restaurant_group_size_l3523_352320

theorem restaurant_group_size :
  let adult_meal_cost : ℕ := 3
  let kids_eat_free : Bool := true
  let num_kids : ℕ := 7
  let total_cost : ℕ := 15
  let num_adults : ℕ := total_cost / adult_meal_cost
  let total_people : ℕ := num_adults + num_kids
  total_people = 12 := by
  sorry

end restaurant_group_size_l3523_352320


namespace bridge_length_l3523_352307

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_sec : ℝ) :
  train_length = 145 →
  train_speed_kmh = 45 →
  crossing_time_sec = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 230 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time_sec) - train_length :=
by sorry

end bridge_length_l3523_352307


namespace find_b_value_l3523_352312

theorem find_b_value (b : ℝ) : (5 : ℝ)^2 + b * 5 - 35 = 0 → b = 2 := by
  sorry

end find_b_value_l3523_352312


namespace zoe_average_speed_l3523_352322

/-- Represents the hiking scenario with Chantal and Zoe -/
structure HikingScenario where
  d : ℝ  -- Represents one-third of the total distance
  chantal_speed1 : ℝ  -- Chantal's speed for the first third
  chantal_speed2 : ℝ  -- Chantal's speed for the rocky part
  chantal_speed3 : ℝ  -- Chantal's speed for descent on rocky part

/-- The theorem stating Zoe's average speed -/
theorem zoe_average_speed (h : HikingScenario) 
  (h_chantal_speed1 : h.chantal_speed1 = 5)
  (h_chantal_speed2 : h.chantal_speed2 = 3)
  (h_chantal_speed3 : h.chantal_speed3 = 4) :
  let total_time := h.d / h.chantal_speed1 + h.d / h.chantal_speed2 + h.d / h.chantal_speed2 + h.d / h.chantal_speed3
  (h.d / total_time) = 60 / 47 := by
  sorry

#check zoe_average_speed

end zoe_average_speed_l3523_352322


namespace jasons_house_paintable_area_l3523_352388

/-- The total area to be painted in multiple identical bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_rooms * paintable_area

/-- Theorem stating the total area to be painted in Jason's house -/
theorem jasons_house_paintable_area :
  total_paintable_area 4 14 11 9 80 = 1480 := by
  sorry

end jasons_house_paintable_area_l3523_352388


namespace matrix_fourth_power_l3523_352340

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_fourth_power :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end matrix_fourth_power_l3523_352340


namespace fermat_primes_totient_divisor_641_l3523_352330

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

end fermat_primes_totient_divisor_641_l3523_352330


namespace medal_distribution_proof_l3523_352343

def total_sprinters : Nat := 10
def american_sprinters : Nat := 4
def medals : Nat := 3

def ways_to_distribute_medals : Nat :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medalists := non_american_sprinters * (non_american_sprinters - 1) * (non_american_sprinters - 2)
  let one_american_medalist := american_sprinters * medals * (non_american_sprinters * (non_american_sprinters - 1))
  no_american_medalists + one_american_medalist

theorem medal_distribution_proof : 
  ways_to_distribute_medals = 480 := by sorry

end medal_distribution_proof_l3523_352343


namespace village_x_decrease_rate_l3523_352311

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

end village_x_decrease_rate_l3523_352311


namespace odd_decreasing_function_range_l3523_352333

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- State the theorem
theorem odd_decreasing_function_range (a : ℝ) 
  (h_odd : is_odd f) 
  (h_decreasing : is_decreasing f) 
  (h_condition : f (2 - a) + f (4 - a) < 0) : 
  a < 3 := by
  sorry

end odd_decreasing_function_range_l3523_352333


namespace belle_collected_97_stickers_l3523_352305

def belle_stickers (carolyn_stickers : ℕ) (difference : ℕ) : ℕ :=
  carolyn_stickers + difference

theorem belle_collected_97_stickers 
  (h1 : belle_stickers 79 18 = 97) : belle_stickers 79 18 = 97 := by
  sorry

end belle_collected_97_stickers_l3523_352305


namespace half_percent_as_repeating_decimal_l3523_352300

theorem half_percent_as_repeating_decimal : 
  (1 / 2 : ℚ) / 100 = 0.00500 := by sorry

end half_percent_as_repeating_decimal_l3523_352300


namespace villages_with_more_knights_count_l3523_352338

/-- The number of villages on the island -/
def total_villages : ℕ := 1000

/-- The number of inhabitants in each village -/
def inhabitants_per_village : ℕ := 99

/-- The total number of knights on the island -/
def total_knights : ℕ := 54054

/-- The number of people in each village who answered there are more knights -/
def more_knights_answers : ℕ := 66

/-- The number of people in each village who answered there are more liars -/
def more_liars_answers : ℕ := 33

/-- The number of villages with more knights than liars -/
def villages_with_more_knights : ℕ := 638

theorem villages_with_more_knights_count :
  villages_with_more_knights = 
    (total_knights - more_liars_answers * total_villages) / 
    (more_knights_answers - more_liars_answers) :=
by sorry

end villages_with_more_knights_count_l3523_352338


namespace largest_divisor_five_consecutive_integers_l3523_352370

def consecutive_integers (n : ℕ) (start : ℤ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

theorem largest_divisor_five_consecutive_integers :
  ∀ start : ℤ, 
  ∃ m : ℕ, m = 240 ∧ 
  (m : ℤ) ∣ (List.prod (consecutive_integers 5 start)) ∧
  ∀ k : ℕ, k > m → ¬((k : ℤ) ∣ (List.prod (consecutive_integers 5 start))) :=
by sorry

end largest_divisor_five_consecutive_integers_l3523_352370


namespace f_properties_l3523_352321

def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem f_properties :
  (∀ x, f 0 x = f 0 (-x)) ∧
  (∀ a, a > 1/2 → ∀ x, f a x ≥ a + 3/4) ∧
  (∀ a, a ≤ -1/2 → ∀ x, f a x ≥ -a + 3/4) ∧
  (∀ a, -1/2 < a ∧ a ≤ 1/2 → ∀ x, f a x ≥ a^2 + 1) :=
by sorry

end f_properties_l3523_352321


namespace polygon_with_40_degree_exterior_angles_has_9_sides_l3523_352348

/-- The number of sides in a polygon where each exterior angle measures 40 degrees -/
def polygon_sides : ℕ :=
  (360 : ℕ) / 40

/-- Theorem: A polygon with exterior angles of 40° has 9 sides -/
theorem polygon_with_40_degree_exterior_angles_has_9_sides :
  polygon_sides = 9 := by
  sorry

end polygon_with_40_degree_exterior_angles_has_9_sides_l3523_352348


namespace olly_shoes_count_l3523_352315

/-- The number of shoes needed for Olly's pets -/
def shoes_needed (num_dogs num_cats num_ferrets : ℕ) : ℕ :=
  4 * (num_dogs + num_cats + num_ferrets)

/-- Theorem: Olly needs 24 shoes for his pets -/
theorem olly_shoes_count : shoes_needed 3 2 1 = 24 := by
  sorry

end olly_shoes_count_l3523_352315


namespace inequality_proof_l3523_352356

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
  (h : a / (1 - x) + b / (1 - y) = 1) : 
  (a * y) ^ (1/3 : ℝ) + (b * x) ^ (1/3 : ℝ) ≤ 1 := by
sorry

end inequality_proof_l3523_352356


namespace percentage_problem_l3523_352372

theorem percentage_problem (P : ℝ) : P = 0.7 ↔ 
  0.8 * 90 = P * 60.00000000000001 + 30 := by
  sorry

end percentage_problem_l3523_352372


namespace investment_return_percentage_l3523_352301

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


end investment_return_percentage_l3523_352301


namespace least_positive_integer_l3523_352359

theorem least_positive_integer (x : ℕ) : x = 6 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → ¬((2*y)^2 + 2*41*(2*y) + 41^2) % 53 = 0) ∧
  ((2*x)^2 + 2*41*(2*x) + 41^2) % 53 = 0 :=
by sorry

end least_positive_integer_l3523_352359


namespace square_plus_n_plus_one_is_odd_l3523_352360

theorem square_plus_n_plus_one_is_odd (n : ℤ) : Odd (n^2 + n + 1) := by
  sorry

end square_plus_n_plus_one_is_odd_l3523_352360


namespace max_of_roots_l3523_352329

theorem max_of_roots (α β γ : ℝ) 
  (sum_eq : α + β + γ = 14)
  (sum_squares_eq : α^2 + β^2 + γ^2 = 84)
  (sum_cubes_eq : α^3 + β^3 + γ^3 = 584) :
  max α (max β γ) = 8 := by
  sorry

end max_of_roots_l3523_352329


namespace range_of_m_l3523_352337

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 := by
  sorry

end range_of_m_l3523_352337


namespace chocolate_cuts_l3523_352389

/-- The minimum number of cuts required to divide a single piece into n pieces -/
def min_cuts (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of cuts to get 24 pieces is 23 -/
theorem chocolate_cuts : min_cuts 24 = 23 := by
  sorry

end chocolate_cuts_l3523_352389


namespace smallest_quotient_l3523_352383

/-- Represents a three-digit number with different non-zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_nonzero : hundreds ≠ 0
  t_nonzero : tens ≠ 0
  o_nonzero : ones ≠ 0
  h_lt_ten : hundreds < 10
  t_lt_ten : tens < 10
  o_lt_ten : ones < 10
  all_different : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

/-- The value of a ThreeDigitNumber -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a ThreeDigitNumber -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The quotient of a ThreeDigitNumber divided by its digit sum -/
def quotient (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digitSum n : Rat)

theorem smallest_quotient :
  ∃ (n : ThreeDigitNumber), ∀ (m : ThreeDigitNumber), quotient n ≤ quotient m ∧ quotient n = 10.5 := by
  sorry

end smallest_quotient_l3523_352383


namespace angle_bisector_length_l3523_352377

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 8 ∧ qr = 15 ∧ pr = 17

-- Define the angle bisector QS
def AngleBisector (P Q R S : ℝ × ℝ) : Prop :=
  let ps := Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2)
  let rs := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  ps / rs = 8 / 15

-- Theorem statement
theorem angle_bisector_length (P Q R S : ℝ × ℝ) :
  Triangle P Q R → AngleBisector P Q R S →
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 4 * Real.sqrt 3272 / 23 :=
by sorry


end angle_bisector_length_l3523_352377


namespace power_multiplication_division_equality_l3523_352328

theorem power_multiplication_division_equality : (12 : ℚ)^2 * 6^3 / 432 = 72 := by sorry

end power_multiplication_division_equality_l3523_352328


namespace cricket_team_size_is_eleven_l3523_352352

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

end cricket_team_size_is_eleven_l3523_352352


namespace couple_ticket_cost_l3523_352394

theorem couple_ticket_cost (single_ticket_cost : ℚ) (total_sales : ℚ) 
  (total_attendance : ℕ) (couple_tickets_sold : ℕ) :
  single_ticket_cost = 20 →
  total_sales = 2280 →
  total_attendance = 128 →
  couple_tickets_sold = 16 →
  ∃ couple_ticket_cost : ℚ,
    couple_ticket_cost = 22.5 ∧
    total_sales = (total_attendance - 2 * couple_tickets_sold) * single_ticket_cost + 
                  couple_tickets_sold * couple_ticket_cost :=
by
  sorry


end couple_ticket_cost_l3523_352394


namespace water_displaced_squared_l3523_352379

/-- The volume of water displaced by a cube submerged in a cylindrical barrel -/
def water_displaced (cube_side : ℝ) (barrel_radius : ℝ) (barrel_height : ℝ) : ℝ :=
  cube_side ^ 3

/-- Theorem: The square of the volume of water displaced by a 10-foot cube
    in a cylindrical barrel is 1,000,000 cubic feet -/
theorem water_displaced_squared :
  let cube_side : ℝ := 10
  let barrel_radius : ℝ := 5
  let barrel_height : ℝ := 15
  (water_displaced cube_side barrel_radius barrel_height) ^ 2 = 1000000 := by
sorry

end water_displaced_squared_l3523_352379


namespace two_true_propositions_l3523_352327

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the parallel and perpendicular relations
def parallel : Plane → Plane → Prop := sorry
def perpendicular : Plane → Plane → Prop := sorry
def parallel_line_plane : Line → Plane → Prop := sorry
def perpendicular_line_plane : Line → Plane → Prop := sorry
def parallel_lines : Line → Line → Prop := sorry
def perpendicular_lines : Line → Line → Prop := sorry

-- Define the original proposition for planes
def original_proposition (α β γ : Plane) : Prop :=
  parallel α β ∧ perpendicular α γ → perpendicular β γ

-- Define the propositions with two planes replaced by lines
def prop_αβ_lines (a b : Line) (γ : Plane) : Prop :=
  parallel_lines a b ∧ perpendicular_line_plane a γ → perpendicular_line_plane b γ

def prop_αγ_lines (a : Line) (β : Plane) (b : Line) : Prop :=
  parallel_line_plane a β ∧ perpendicular_lines a b → perpendicular_line_plane b β

def prop_βγ_lines (α : Plane) (a b : Line) : Prop :=
  parallel_line_plane a α ∧ perpendicular_line_plane α b → perpendicular_lines a b

-- The main theorem
theorem two_true_propositions :
  ∃ (α β γ : Plane),
    original_proposition α β γ = true ∧
    (∀ (a b : Line),
      (prop_αβ_lines a b γ = true ∧ prop_αγ_lines a β b = false ∧ prop_βγ_lines α a b = true) ∨
      (prop_αβ_lines a b γ = true ∧ prop_αγ_lines a β b = true ∧ prop_βγ_lines α a b = false) ∨
      (prop_αβ_lines a b γ = false ∧ prop_αγ_lines a β b = true ∧ prop_βγ_lines α a b = true)) :=
sorry

end two_true_propositions_l3523_352327


namespace indeterminate_b_value_l3523_352353

theorem indeterminate_b_value (a b c d : ℝ) : 
  a > b ∧ b > c ∧ c > d → 
  (a + b + c + d) / 4 = 12.345 → 
  ¬(∀ x : ℝ, x = b → (x > 12.345 ∨ x < 12.345 ∨ x = 12.345)) :=
by sorry

end indeterminate_b_value_l3523_352353


namespace existence_of_strictly_decreasing_function_with_inequality_l3523_352368

/-- A strictly decreasing function from (0, +∞) to (0, +∞) -/
def StrictlyDecreasingPositiveFunction :=
  {g : ℝ → ℝ | ∀ x y, 0 < x → 0 < y → x < y → g y < g x}

theorem existence_of_strictly_decreasing_function_with_inequality
  (k : ℝ) (h_k : 0 < k) :
  (∃ g : ℝ → ℝ, g ∈ StrictlyDecreasingPositiveFunction ∧
    ∀ x, 0 < x → 0 < g x ∧ g x ≥ k * g (x + g x)) ↔ k ≤ 1 := by
  sorry

end existence_of_strictly_decreasing_function_with_inequality_l3523_352368


namespace anthonys_pets_l3523_352354

theorem anthonys_pets (initial_pets : ℕ) (lost_pets : ℕ) (final_pets : ℕ) :
  initial_pets = 16 →
  lost_pets = 6 →
  final_pets = 8 →
  (initial_pets - lost_pets - final_pets : ℚ) / (initial_pets - lost_pets) = 1/5 := by
  sorry

end anthonys_pets_l3523_352354


namespace seniors_physical_books_l3523_352314

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

end seniors_physical_books_l3523_352314


namespace sock_pairs_theorem_l3523_352319

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

end sock_pairs_theorem_l3523_352319


namespace sum_seven_smallest_multiples_of_12_l3523_352336

theorem sum_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (fun i => 12 * (i + 1)) = 336 := by
  sorry

end sum_seven_smallest_multiples_of_12_l3523_352336


namespace sin_330_degrees_l3523_352347

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l3523_352347
