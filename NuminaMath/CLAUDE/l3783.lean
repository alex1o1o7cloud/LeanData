import Mathlib

namespace length_CD_l3783_378307

theorem length_CD (AB D C : ℝ) : 
  AB = 48 →                 -- Length of AB is 48
  D = AB / 3 →              -- AD is 1/3 of AB
  C = AB / 2 →              -- C is the midpoint of AB
  C - D = 8 :=              -- Length of CD is 8
by
  sorry


end length_CD_l3783_378307


namespace factor_tree_value_l3783_378388

-- Define the variables
def W : ℕ := 7
def Y : ℕ := 7 * 11
def Z : ℕ := 13 * W
def X : ℕ := Y * Z

-- State the theorem
theorem factor_tree_value : X = 7007 := by
  sorry

end factor_tree_value_l3783_378388


namespace price_decrease_proof_l3783_378397

theorem price_decrease_proof (original_price : ℝ) (decrease_percentage : ℝ) (new_price : ℝ) :
  decrease_percentage = 24 →
  new_price = 421.05263157894734 →
  new_price = original_price * (1 - decrease_percentage / 100) :=
by
  sorry

#eval 421.05263157894734 -- To show the exact value used in the problem

end price_decrease_proof_l3783_378397


namespace storeroom_contains_912_blocks_l3783_378323

/-- Calculates the number of blocks in a rectangular storeroom with given dimensions and wall thickness -/
def storeroom_blocks (length width height wall_thickness : ℕ) : ℕ :=
  let total_volume := length * width * height
  let internal_length := length - 2 * wall_thickness
  let internal_width := width - 2 * wall_thickness
  let internal_height := height - wall_thickness
  let internal_volume := internal_length * internal_width * internal_height
  total_volume - internal_volume

/-- Theorem stating that a storeroom with given dimensions contains 912 blocks -/
theorem storeroom_contains_912_blocks :
  storeroom_blocks 15 12 8 2 = 912 := by
  sorry

#eval storeroom_blocks 15 12 8 2

end storeroom_contains_912_blocks_l3783_378323


namespace rectangle_diagonal_length_l3783_378312

/-- The length of the diagonal of a rectangle with specific properties -/
theorem rectangle_diagonal_length : ∀ (a b d : ℝ), 
  a > 0 → 
  b = 2 * a → 
  a = 40 * Real.sqrt 2 → 
  d^2 = a^2 + b^2 → 
  d = 160 :=
by
  sorry

end rectangle_diagonal_length_l3783_378312


namespace stratified_sample_theorem_l3783_378366

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  total_population : ℕ
  male_population : ℕ
  female_population : ℕ
  female_sample : ℕ
  male_sample : ℕ

/-- Checks if a stratified sample is valid according to the stratified sampling principle -/
def is_valid_stratified_sample (s : StratifiedSample) : Prop :=
  s.female_population * s.male_sample = s.male_population * s.female_sample

theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_population = 680)
  (h2 : s.male_population = 360)
  (h3 : s.female_population = 320)
  (h4 : s.female_sample = 16)
  (h5 : is_valid_stratified_sample s) :
  s.male_sample = 18 := by
  sorry

#check stratified_sample_theorem

end stratified_sample_theorem_l3783_378366


namespace sons_age_is_correct_l3783_378365

/-- The age of the son -/
def sons_age : ℕ := 23

/-- The age of the father -/
def fathers_age : ℕ := sons_age + 25

theorem sons_age_is_correct : 
  (fathers_age + 2 = 2 * (sons_age + 2)) ∧ 
  (fathers_age = sons_age + 25) ∧ 
  (sons_age = 23) := by
  sorry

end sons_age_is_correct_l3783_378365


namespace arithmetic_sequence_property_l3783_378308

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_is_term (a : ℕ → ℕ) : Prop :=
  ∀ p s, ∃ t, a p + a s = a t

theorem arithmetic_sequence_property (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d →
  a 1 = 12 →
  d > 0 →
  sum_is_term a →
  d = 6 ∨ d = 3 ∨ d = 2 ∨ d = 1 →
  d = 6 :=
sorry

end arithmetic_sequence_property_l3783_378308


namespace max_displayed_games_l3783_378373

/-- Represents the number of games that can be displayed for each genre -/
structure DisplayedGames where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

/-- Represents the shelf capacity for each genre -/
structure ShelfCapacity where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

/-- Represents the total number of games for each genre -/
structure TotalGames where
  action : ℕ
  adventure : ℕ
  simulation : ℕ

def store_promotion : ℕ := 10

def total_games : TotalGames :=
  { action := 73, adventure := 51, simulation := 39 }

def shelf_capacity : ShelfCapacity :=
  { action := 60, adventure := 45, simulation := 35 }

def displayed_games (t : TotalGames) (s : ShelfCapacity) : DisplayedGames :=
  { action := min (t.action - store_promotion) s.action + store_promotion,
    adventure := min (t.adventure - store_promotion) s.adventure + store_promotion,
    simulation := min (t.simulation - store_promotion) s.simulation + store_promotion }

def total_displayed (d : DisplayedGames) : ℕ :=
  d.action + d.adventure + d.simulation

theorem max_displayed_games :
  total_displayed (displayed_games total_games shelf_capacity) = 160 :=
by sorry

end max_displayed_games_l3783_378373


namespace simple_interest_rate_calculation_l3783_378311

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R > 0 ∧ R / 100 * P * 10 = 4/5 * P ∧ R = 8 := by
  sorry

end simple_interest_rate_calculation_l3783_378311


namespace fraction_simplification_l3783_378352

theorem fraction_simplification :
  (240 : ℚ) / 20 * 6 / 150 * 12 / 5 = 48 / 125 := by sorry

end fraction_simplification_l3783_378352


namespace power_of_square_l3783_378339

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_square_l3783_378339


namespace smallest_angle_solution_l3783_378304

theorem smallest_angle_solution (y : Real) : 
  (∀ θ : Real, θ > 0 ∧ θ < y → 10 * Real.sin θ * Real.cos θ ^ 3 - 10 * Real.sin θ ^ 3 * Real.cos θ ≠ Real.sqrt 2) ∧
  (10 * Real.sin y * Real.cos y ^ 3 - 10 * Real.sin y ^ 3 * Real.cos y = Real.sqrt 2) →
  y = 11.25 * π / 180 := by
  sorry

end smallest_angle_solution_l3783_378304


namespace storage_house_blocks_l3783_378344

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular prism -/
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the specifications of the storage house -/
structure StorageHouse where
  outerDimensions : Dimensions
  wallThickness : ℕ

/-- Calculates the inner dimensions of the storage house -/
def innerDimensions (s : StorageHouse) : Dimensions :=
  { length := s.outerDimensions.length - 2 * s.wallThickness,
    width := s.outerDimensions.width - 2 * s.wallThickness,
    height := s.outerDimensions.height - s.wallThickness }

/-- Calculates the number of blocks needed for the storage house -/
def blocksNeeded (s : StorageHouse) : ℕ :=
  volume s.outerDimensions - volume (innerDimensions s)

theorem storage_house_blocks :
  let s : StorageHouse :=
    { outerDimensions := { length := 15, width := 12, height := 8 },
      wallThickness := 2 }
  blocksNeeded s = 912 := by sorry

end storage_house_blocks_l3783_378344


namespace altitude_polynomial_l3783_378351

/-- Given a cubic polynomial with rational coefficients whose roots are the side lengths of a triangle,
    the altitudes of this triangle are roots of a polynomial of sixth degree with rational coefficients. -/
theorem altitude_polynomial (a b c d : ℚ) (r₁ r₂ r₃ : ℝ) :
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) →
  (r₁ + r₂ > r₃ ∧ r₂ + r₃ > r₁ ∧ r₃ + r₁ > r₂) →
  ∃ (p q s t u v w : ℚ),
    ∀ h₁ h₂ h₃ : ℝ,
      (h₁ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₁ ∧
       h₂ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₂ ∧
       h₃ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₃) →
      p * h₁^6 + q * h₁^5 + s * h₁^4 + t * h₁^3 + u * h₁^2 + v * h₁ + w = 0 ∧
      p * h₂^6 + q * h₂^5 + s * h₂^4 + t * h₂^3 + u * h₂^2 + v * h₂ + w = 0 ∧
      p * h₃^6 + q * h₃^5 + s * h₃^4 + t * h₃^3 + u * h₃^2 + v * h₃ + w = 0 :=
by sorry

end altitude_polynomial_l3783_378351


namespace periodic_function_value_l3783_378302

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2 * Real.pi) = f x

theorem periodic_function_value 
  (f : ℝ → ℝ) 
  (h1 : periodic_function f) 
  (h2 : f 0 = 0) : 
  f (4 * Real.pi) = 0 := by
sorry

end periodic_function_value_l3783_378302


namespace chessboard_division_theorem_l3783_378376

/-- Represents a 6x6 chessboard --/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- Represents a 2x1 domino on the chessboard --/
structure Domino where
  x : Fin 6
  y : Fin 6
  horizontal : Bool

/-- A configuration of dominoes on the chessboard --/
def DominoConfiguration := List Domino

/-- Checks if a given line (horizontal or vertical) intersects any domino --/
def lineIntersectsDomino (line : Nat) (horizontal : Bool) (config : DominoConfiguration) : Bool :=
  sorry

/-- The main theorem --/
theorem chessboard_division_theorem (config : DominoConfiguration) :
  config.length = 18 → ∃ (line : Nat) (horizontal : Bool),
    line < 6 ∧ ¬lineIntersectsDomino line horizontal config :=
  sorry

end chessboard_division_theorem_l3783_378376


namespace download_time_360GB_50MBps_l3783_378363

/-- Calculates the download time in hours for a given program size and download speed -/
def downloadTime (programSizeGB : ℕ) (downloadSpeedMBps : ℕ) : ℚ :=
  let programSizeMB := programSizeGB * 1000
  let downloadTimeSeconds := programSizeMB / downloadSpeedMBps
  downloadTimeSeconds / 3600

/-- Proves that downloading a 360 GB program at 50 MB/s takes 2 hours -/
theorem download_time_360GB_50MBps :
  downloadTime 360 50 = 2 := by
  sorry

end download_time_360GB_50MBps_l3783_378363


namespace projection_implies_coplanar_and_parallel_l3783_378332

-- Define a type for 3D points
def Point3D := ℝ × ℝ × ℝ

-- Define a type for 2D points (projections)
def Point2D := ℝ × ℝ

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  d : ℝ

-- Define a projection function
def project (p : Point3D) (plane : Plane) : Point2D :=
  sorry

-- Define a predicate for points being on a line
def onLine (points : List Point2D) : Prop :=
  sorry

-- Define a predicate for points being coplanar
def coplanar (points : List Point3D) : Prop :=
  sorry

-- Define a predicate for points being parallel
def parallel (points : List Point3D) : Prop :=
  sorry

-- The main theorem
theorem projection_implies_coplanar_and_parallel 
  (points : List Point3D) (plane : Plane) :
  onLine (points.map (λ p => project p plane)) →
  coplanar points ∧ parallel points :=
sorry

end projection_implies_coplanar_and_parallel_l3783_378332


namespace hexagon_properties_l3783_378321

/-- A regular hexagon with diagonals -/
structure RegularHexagonWithDiagonals where
  /-- The area of the regular hexagon -/
  area : ℝ
  /-- The hexagon is regular -/
  is_regular : Bool
  /-- All diagonals are drawn -/
  diagonals_drawn : Bool

/-- The number of parts the hexagon is divided into by its diagonals -/
def num_parts (h : RegularHexagonWithDiagonals) : ℕ := sorry

/-- The area of the new regular hexagon formed by combining all quadrilateral parts -/
def new_hexagon_area (h : RegularHexagonWithDiagonals) : ℝ := sorry

/-- Theorem about the properties of a regular hexagon with diagonals -/
theorem hexagon_properties (h : RegularHexagonWithDiagonals) 
  (h_area : h.area = 144)
  (h_regular : h.is_regular = true)
  (h_diagonals : h.diagonals_drawn = true) :
  num_parts h = 24 ∧ new_hexagon_area h = 48 := by sorry

end hexagon_properties_l3783_378321


namespace braiding_time_for_dance_team_l3783_378378

/-- Calculates the time in minutes to braid dancers' hair -/
def braiding_time (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℕ :=
  (num_dancers * braids_per_dancer * seconds_per_braid) / 60

/-- Proves that braiding 8 dancers' hair with 5 braids each, taking 30 seconds per braid, results in 20 minutes total -/
theorem braiding_time_for_dance_team : braiding_time 8 5 30 = 20 := by
  sorry

end braiding_time_for_dance_team_l3783_378378


namespace dad_vacuum_time_l3783_378305

theorem dad_vacuum_time (downstairs upstairs : ℕ) : 
  upstairs = 2 * downstairs + 5 →
  downstairs + upstairs = 38 →
  upstairs = 27 := by
sorry

end dad_vacuum_time_l3783_378305


namespace salary_problem_l3783_378327

theorem salary_problem (total : ℝ) (a_spend_percent : ℝ) (b_spend_percent : ℝ)
  (h_total : total = 7000)
  (h_a_spend : a_spend_percent = 95)
  (h_b_spend : b_spend_percent = 85)
  (h_equal_savings : (100 - a_spend_percent) * a_salary = (100 - b_spend_percent) * (total - a_salary)) :
  a_salary = 5250 :=
by
  sorry

#check salary_problem

end salary_problem_l3783_378327


namespace stadium_length_in_feet_l3783_378342

/-- Converts yards to feet using the standard conversion factor. -/
def yards_to_feet (yards : ℕ) : ℕ := yards * 3

/-- The length of the sports stadium in yards. -/
def stadium_length_yards : ℕ := 61

theorem stadium_length_in_feet :
  yards_to_feet stadium_length_yards = 183 := by
  sorry

end stadium_length_in_feet_l3783_378342


namespace count_valid_numbers_l3783_378374

/-- The number of n-digit numbers formed using the digits 1, 2, and 3, where each digit is used at least once -/
def valid_numbers (n : ℕ) : ℕ :=
  3^n - 3 * 2^n + 3

/-- Theorem stating that for n ≥ 3, the number of n-digit numbers formed using the digits 1, 2, and 3, 
    where each digit is used at least once, is equal to 3^n - 3 * 2^n + 3 -/
theorem count_valid_numbers (n : ℕ) (h : n ≥ 3) : 
  (valid_numbers n) = (3^n - 3 * 2^n + 3) := by
  sorry


end count_valid_numbers_l3783_378374


namespace rectangle_circle_area_ratio_l3783_378381

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 3 * w) : 
  (l * w) / (Real.pi * r^2) = 3 * Real.pi / 16 := by
  sorry

end rectangle_circle_area_ratio_l3783_378381


namespace complex_calculation_l3783_378316

theorem complex_calculation : (1 - Complex.I) - (-3 + 2 * Complex.I) + (4 - 6 * Complex.I) = 8 - 9 * Complex.I := by
  sorry

end complex_calculation_l3783_378316


namespace percentage_of_democrats_l3783_378372

theorem percentage_of_democrats (D R : ℝ) : 
  D + R = 100 →
  0.7 * D + 0.2 * R = 50 →
  D = 60 := by
sorry

end percentage_of_democrats_l3783_378372


namespace binomial_coefficient_21_12_l3783_378338

theorem binomial_coefficient_21_12 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 12 = 125970) →
  (Nat.choose 21 13 = 203490) →
  (Nat.choose 21 12 = 125970) := by
  sorry

end binomial_coefficient_21_12_l3783_378338


namespace solve_invitations_l3783_378355

def invitations_problem (I : ℝ) : Prop :=
  let rsvp_rate : ℝ := 0.9
  let show_up_rate : ℝ := 0.8
  let no_gift_attendees : ℕ := 10
  let thank_you_cards : ℕ := 134
  
  (rsvp_rate * show_up_rate * I - no_gift_attendees : ℝ) = thank_you_cards

theorem solve_invitations : ∃ I : ℝ, invitations_problem I ∧ I = 200 := by
  sorry

end solve_invitations_l3783_378355


namespace triangle_equilateral_proof_l3783_378343

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if certain conditions are met, the triangle is equilateral with A = π/3. -/
theorem triangle_equilateral_proof (a b c A B C : ℝ) : 
  0 < A ∧ A < π →  -- Angle A is between 0 and π
  0 < B ∧ B < π →  -- Angle B is between 0 and π
  0 < C ∧ C < π →  -- Angle C is between 0 and π
  A + B + C = π →  -- Sum of angles in a triangle
  2 * a * Real.cos B = 2 * c - b →  -- Given condition
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 4 →  -- Area condition
  a = Real.sqrt 3 →  -- Given side length
  A = π/3 ∧ a = b ∧ b = c := by sorry

end triangle_equilateral_proof_l3783_378343


namespace remainder_sum_l3783_378324

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9 = 3) := by
  sorry

end remainder_sum_l3783_378324


namespace integral_f_equals_pi_over_four_l3783_378331

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + Real.tan (Real.sqrt 2 * x))

theorem integral_f_equals_pi_over_four :
  ∫ x in (0)..(Real.pi / 2), f x = Real.pi / 4 := by sorry

end integral_f_equals_pi_over_four_l3783_378331


namespace a_periodic_with_period_5_l3783_378369

/-- The sequence a_n defined as 6^n mod 100 -/
def a (n : ℕ) : ℕ := (6^n) % 100

/-- The period of the sequence a_n -/
def period : ℕ := 5

theorem a_periodic_with_period_5 :
  (∀ n ≥ 2, a (n + period) = a n) ∧
  (∀ k < period, ∃ m ≥ 2, a (m + k) ≠ a m) :=
sorry

end a_periodic_with_period_5_l3783_378369


namespace other_lateral_side_length_l3783_378386

/-- A trapezoid with the property that a line through the midpoint of one lateral side
    divides it into two quadrilaterals, each with an inscribed circle -/
structure SpecialTrapezoid where
  /-- Length of one base -/
  a : ℝ
  /-- Length of the other base -/
  b : ℝ
  /-- The trapezoid has the special property -/
  has_special_property : Bool

/-- The length of the other lateral side in a special trapezoid -/
def other_lateral_side (t : SpecialTrapezoid) : ℝ :=
  t.a + t.b

theorem other_lateral_side_length (t : SpecialTrapezoid) 
  (h : t.has_special_property = true) : 
  other_lateral_side t = t.a + t.b := by
  sorry

end other_lateral_side_length_l3783_378386


namespace irrational_number_existence_l3783_378325

theorem irrational_number_existence : ∃ α : ℝ, (α > 1) ∧ (Irrational α) ∧
  (∀ n : ℕ, n ≥ 1 → (⌊α^n⌋ : ℤ) % 2017 = 0) := by
  sorry

end irrational_number_existence_l3783_378325


namespace rectangle_circle_union_area_l3783_378356

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let circle_radius : ℝ := 12
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := π * circle_radius^2
  let quarter_circle_area : ℝ := circle_area / 4
  rectangle_area + (circle_area - quarter_circle_area) = 96 + 108 * π := by
sorry

end rectangle_circle_union_area_l3783_378356


namespace function_always_negative_l3783_378322

theorem function_always_negative
  (f : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, (2 - x) * f x + x * deriv f x < 0) :
  ∀ x : ℝ, f x < 0 :=
by sorry

end function_always_negative_l3783_378322


namespace parallel_vectors_m_value_l3783_378391

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, m)
  parallel a b → m = -4 :=
by
  sorry

end parallel_vectors_m_value_l3783_378391


namespace train_distance_example_l3783_378375

/-- The total distance traveled by a train given its speed and time -/
def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train traveling at 85 km/h for 4 hours covers 340 km -/
theorem train_distance_example : train_distance 85 4 = 340 := by
  sorry

end train_distance_example_l3783_378375


namespace pen_price_correct_max_pens_correct_l3783_378399

-- Define the original price of a pen
def original_pen_price : ℝ := 4

-- Define the discount rate for pens in the first part
def discount_rate : ℝ := 0.1

-- Define the total budget
def budget : ℝ := 360

-- Define the number of additional pens that can be bought after discount
def additional_pens : ℕ := 10

-- Define the total number of items to be purchased
def total_items : ℕ := 80

-- Define the original price of a pencil case
def pencil_case_price : ℝ := 10

-- Define the discount rate for both items in the second part
def discount_rate_2 : ℝ := 0.2

-- Define the minimum total purchase amount
def min_purchase_amount : ℝ := 400

theorem pen_price_correct :
  budget / original_pen_price + additional_pens = budget / (original_pen_price * (1 - discount_rate)) :=
sorry

theorem max_pens_correct :
  ∀ y : ℕ, y ≤ 50 →
  y ≤ total_items →
  min_purchase_amount ≤ original_pen_price * (1 - discount_rate_2) * y + pencil_case_price * (1 - discount_rate_2) * (total_items - y) :=
sorry

#check pen_price_correct
#check max_pens_correct

end pen_price_correct_max_pens_correct_l3783_378399


namespace average_work_hours_l3783_378328

theorem average_work_hours (total_people : ℕ) (people_on_duty : ℕ) (hours_per_day : ℕ) :
  total_people = 8 →
  people_on_duty = 3 →
  hours_per_day = 24 →
  (hours_per_day * people_on_duty : ℚ) / total_people = 9 := by
sorry

end average_work_hours_l3783_378328


namespace interval_of_decrease_l3783_378354

/-- The function f(x) = -x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -2*x + 4

theorem interval_of_decrease (x : ℝ) :
  x ≥ 2 → (∀ y, y > x → f y < f x) :=
by sorry

end interval_of_decrease_l3783_378354


namespace single_filter_price_l3783_378377

/-- The price of a camera lens filter kit containing 5 filters -/
def kit_price : ℝ := 87.50

/-- The price of the first type of filter -/
def filter1_price : ℝ := 16.45

/-- The price of the second type of filter -/
def filter2_price : ℝ := 14.05

/-- The discount rate when purchasing the kit -/
def discount_rate : ℝ := 0.08

/-- The number of filters of the first type -/
def num_filter1 : ℕ := 2

/-- The number of filters of the second type -/
def num_filter2 : ℕ := 2

/-- The number of filters of the unknown type -/
def num_filter3 : ℕ := 1

/-- The total number of filters in the kit -/
def total_filters : ℕ := num_filter1 + num_filter2 + num_filter3

theorem single_filter_price (x : ℝ) : 
  (num_filter1 : ℝ) * filter1_price + (num_filter2 : ℝ) * filter2_price + (num_filter3 : ℝ) * x = 
  kit_price / (1 - discount_rate) → x = 34.11 := by
  sorry

end single_filter_price_l3783_378377


namespace mothers_birthday_knowledge_l3783_378368

/-- Represents the distribution of students' knowledge about their parents' birthdays -/
structure BirthdayKnowledge where
  total : ℕ
  only_father : ℕ
  only_mother : ℕ
  both_parents : ℕ
  neither_parent : ℕ

/-- Theorem stating that 22 students know their mother's birthday -/
theorem mothers_birthday_knowledge (bk : BirthdayKnowledge) 
  (h1 : bk.total = 40)
  (h2 : bk.only_father = 10)
  (h3 : bk.only_mother = 12)
  (h4 : bk.both_parents = 22)
  (h5 : bk.neither_parent = 26)
  (h6 : bk.total = bk.only_father + bk.only_mother + bk.both_parents + bk.neither_parent) :
  bk.only_mother + bk.both_parents = 22 := by
  sorry

end mothers_birthday_knowledge_l3783_378368


namespace hyperbola_parameters_l3783_378310

/-- Theorem: For a hyperbola with given conditions, a = 1 and b = 4 -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y, x^2 / a - y^2 / b = 1) →  -- Hyperbola equation
  (∃ k, ∀ x y, 2*x + y = 0 → y = k*x) →  -- One asymptote
  (∃ x y, x^2 + y^2 = 5 ∧ y = 0) →  -- One focus
  a = 1 ∧ b = 4 := by
sorry

end hyperbola_parameters_l3783_378310


namespace range_of_m_l3783_378349

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * m * x + 9 ≥ 0) → 
  m ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by
  sorry

end range_of_m_l3783_378349


namespace quadratic_equation_properties_l3783_378353

theorem quadratic_equation_properties :
  ∀ s t : ℝ, 2 * s^2 + 3 * s - 1 = 0 → 2 * t^2 + 3 * t - 1 = 0 → s ≠ t →
  (s + t = -3/2) ∧
  (s * t = -1/2) ∧
  (s^2 + t^2 = 13/4) ∧
  (|1/s - 1/t| = Real.sqrt 17) :=
by sorry

end quadratic_equation_properties_l3783_378353


namespace max_increase_year_1998_l3783_378318

def sales : Fin 11 → ℝ
  | 0 => 3.0
  | 1 => 4.5
  | 2 => 5.1
  | 3 => 7.0
  | 4 => 8.5
  | 5 => 9.7
  | 6 => 10.7
  | 7 => 12.0
  | 8 => 13.2
  | 9 => 13.7
  | 10 => 7.5

def year_increase (i : Fin 10) : ℝ :=
  sales (i.succ) - sales i

theorem max_increase_year_1998 :
  ∃ i : Fin 10, (i.val + 1995 = 1998) ∧
    ∀ j : Fin 10, year_increase j ≤ year_increase i :=
by sorry

end max_increase_year_1998_l3783_378318


namespace complex_fraction_simplification_l3783_378345

theorem complex_fraction_simplification :
  (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end complex_fraction_simplification_l3783_378345


namespace circle_equation_l3783_378301

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : ℝ := x - y - 1
def line2 (x y : ℝ) : ℝ := 4*x + 3*y + 14
def line3 (x y : ℝ) : ℝ := 3*x + 4*y + 10

-- State the theorem
theorem circle_equation (C : Circle) :
  (∀ x y, line1 x y = 0 → x = C.center.1 ∧ y = C.center.2) →
  (∃ x y, line2 x y = 0 ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) →
  (∃ x1 y1 x2 y2, line3 x1 y1 = 0 ∧ line3 x2 y2 = 0 ∧
    (x1 - C.center.1)^2 + (y1 - C.center.2)^2 = C.radius^2 ∧
    (x2 - C.center.1)^2 + (y2 - C.center.2)^2 = C.radius^2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 36) →
  C.center = (2, 1) ∧ C.radius = 5 :=
by sorry

end circle_equation_l3783_378301


namespace expression_simplification_l3783_378398

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 4) * (9 * p - 12) = 89 * p - 84 := by
  sorry

end expression_simplification_l3783_378398


namespace three_solutions_cosine_sine_equation_l3783_378389

theorem three_solutions_cosine_sine_equation :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, 0 < x ∧ x < 3 * Real.pi ∧ 3 * (Real.cos x)^2 + 2 * (Real.sin x)^2 = 2) ∧
    s.card = 3 := by
  sorry

end three_solutions_cosine_sine_equation_l3783_378389


namespace shopping_cost_l3783_378336

/-- The cost of items in a shopping mall with discount --/
theorem shopping_cost (tshirt_cost pants_cost shoe_cost : ℝ) 
  (h1 : tshirt_cost = 20)
  (h2 : pants_cost = 80)
  (h3 : (4 * tshirt_cost + 3 * pants_cost + 2 * shoe_cost) * 0.9 = 558) :
  shoe_cost = 150 := by
sorry

end shopping_cost_l3783_378336


namespace negation_equivalence_l3783_378382

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Child : U → Prop)
variable (CarefulInvestor : U → Prop)
variable (RecklessInvestor : U → Prop)

-- Define the statements
def AllChildrenAreCareful : Prop := ∀ x, Child x → CarefulInvestor x
def AtLeastOneChildIsReckless : Prop := ∃ x, Child x ∧ RecklessInvestor x

-- The theorem to prove
theorem negation_equivalence : 
  AtLeastOneChildIsReckless U Child RecklessInvestor ↔ 
  ¬(AllChildrenAreCareful U Child CarefulInvestor) :=
sorry

-- Additional assumption: being reckless is the opposite of being careful
axiom reckless_careful_opposite : 
  ∀ x, RecklessInvestor x ↔ ¬(CarefulInvestor x)

end negation_equivalence_l3783_378382


namespace quotient_in_third_quadrant_l3783_378306

/-- Given complex numbers z₁ and z₂ where z₁ = 1 - 2i and the points corresponding to z₁ and z₂ 
    are symmetric about the imaginary axis, the point corresponding to z₂/z₁ lies in the third 
    quadrant of the complex plane. -/
theorem quotient_in_third_quadrant (z₁ z₂ : ℂ) 
    (h₁ : z₁ = 1 - 2*I) 
    (h₂ : z₂.re = -z₁.re ∧ z₂.im = z₁.im) : 
    (z₂ / z₁).re < 0 ∧ (z₂ / z₁).im < 0 := by
  sorry

end quotient_in_third_quadrant_l3783_378306


namespace ellipse_circle_tangent_l3783_378395

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A circle with radius r -/
structure Circle where
  r : ℝ
  h : r > 0

/-- The theorem statement -/
theorem ellipse_circle_tangent (C : Ellipse) (O : Circle) :
  C.a = 2 * Real.sqrt 2 →  -- Left vertex at (-2√2, 0)
  O.r = 2 →  -- Circle equation: x² + y² = 4
  (∃ F : ℝ × ℝ, F.1 = -Real.sqrt 2 ∧ F.2 = 0 ∧
    ∃ A B : ℝ × ℝ, 
      -- A and B are on the circle
      (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧
      -- Line AB passes through F
      (B.2 - A.2) * (F.1 - A.1) = (F.2 - A.2) * (B.1 - A.1)) →
  C.a^2 + C.b^2 = 14 := by
sorry

end ellipse_circle_tangent_l3783_378395


namespace sandcastle_height_difference_l3783_378394

/-- The height difference between Janet's sandcastle and her sister's sandcastle --/
def height_difference : ℝ :=
  let janet_height : ℝ := 3.6666666666666665
  let sister_height : ℝ := 2.3333333333333335
  janet_height - sister_height

/-- Theorem stating that the height difference is 1.333333333333333 feet --/
theorem sandcastle_height_difference :
  height_difference = 1.333333333333333 := by sorry

end sandcastle_height_difference_l3783_378394


namespace rational_equation_implication_l3783_378346

theorem rational_equation_implication (a b : ℚ) 
  (h : Real.sqrt (a + 4) + (b - 2)^2 = 0) : a - b = -6 := by
  sorry

end rational_equation_implication_l3783_378346


namespace range_of_f_set_where_g_less_than_f_l3783_378314

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + x
def g (x : ℝ) : ℝ := |x + 1|

-- Statement for the range of f
theorem range_of_f : Set.range f = Set.Ici 2 := by sorry

-- Statement for the set where g(x) < f(x)
theorem set_where_g_less_than_f : 
  {x : ℝ | g x < f x} = Set.union (Set.Ioo (-3) 1) (Set.Ioi 3) := by sorry

end range_of_f_set_where_g_less_than_f_l3783_378314


namespace triangle_max_perimeter_l3783_378358

theorem triangle_max_perimeter (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  ∃ (n : ℕ), n = 62 ∧ ∀ (s : ℝ), s > 0 → a + s > b → b + s > a → n > a + b + s :=
by sorry

end triangle_max_perimeter_l3783_378358


namespace total_books_combined_l3783_378393

theorem total_books_combined (keith_books jason_books amanda_books sophie_books : ℕ)
  (h1 : keith_books = 20)
  (h2 : jason_books = 21)
  (h3 : amanda_books = 15)
  (h4 : sophie_books = 30) :
  keith_books + jason_books + amanda_books + sophie_books = 86 := by
sorry

end total_books_combined_l3783_378393


namespace room_width_calculation_l3783_378360

/-- Given a room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end room_width_calculation_l3783_378360


namespace condition_equivalence_l3783_378367

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem statement
theorem condition_equivalence (a b : ℝ) :
  (a + b > 0) ↔ (f a + f b > 0) := by
  sorry

end condition_equivalence_l3783_378367


namespace trip_duration_l3783_378348

/-- A car trip with varying speeds -/
structure CarTrip where
  totalTime : ℝ
  averageSpeed : ℝ

/-- The conditions of the car trip -/
def tripConditions (trip : CarTrip) : Prop :=
  ∃ (additionalTime : ℝ),
    trip.totalTime = 4 + additionalTime ∧
    50 * 4 + 80 * additionalTime = 65 * trip.totalTime ∧
    trip.averageSpeed = 65

/-- The theorem stating that the trip duration is 8 hours -/
theorem trip_duration (trip : CarTrip) 
    (h : tripConditions trip) : trip.totalTime = 8 := by
  sorry

#check trip_duration

end trip_duration_l3783_378348


namespace pet_store_siamese_cats_l3783_378361

theorem pet_store_siamese_cats 
  (total_cats : ℕ) 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (sold_cats : ℕ) 
  (remaining_cats : ℕ) :
  house_cats = 49 →
  sold_cats = 19 →
  remaining_cats = 45 →
  total_cats = siamese_cats + house_cats →
  total_cats = remaining_cats + sold_cats →
  siamese_cats = 15 := by
sorry

end pet_store_siamese_cats_l3783_378361


namespace factorial_divisibility_l3783_378326

theorem factorial_divisibility (p : ℕ) (h : Prime p) : 
  (Nat.factorial (p^2)) % (Nat.factorial p)^(p+1) = 0 := by
  sorry

end factorial_divisibility_l3783_378326


namespace william_tax_is_800_l3783_378362

/-- Represents the farm tax system in a village -/
structure FarmTaxSystem where
  total_tax : ℝ
  taxable_land_percentage : ℝ
  william_land_percentage : ℝ

/-- Calculates the farm tax paid by Mr. William -/
def william_tax (system : FarmTaxSystem) : ℝ :=
  system.total_tax * system.william_land_percentage

/-- Theorem stating that Mr. William's farm tax is $800 -/
theorem william_tax_is_800 (system : FarmTaxSystem) 
  (h1 : system.total_tax = 5000)
  (h2 : system.taxable_land_percentage = 0.6)
  (h3 : system.william_land_percentage = 0.16) : 
  william_tax system = 800 := by
  sorry


end william_tax_is_800_l3783_378362


namespace winter_holiday_activities_l3783_378340

theorem winter_holiday_activities (total : ℕ) (skating : ℕ) (skiing : ℕ) (both : ℕ) :
  total = 30 →
  skating = 20 →
  skiing = 9 →
  both = 5 →
  total - (skating + skiing - both) = 6 :=
by sorry

end winter_holiday_activities_l3783_378340


namespace solve_for_y_l3783_378334

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end solve_for_y_l3783_378334


namespace integer_solutions_of_equation_l3783_378364

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 + x = y^4 + y^3 + y^2 + y ↔
    (x = 0 ∧ y = -1) ∨
    (x = -1 ∧ y = -1) ∨
    (x = 0 ∧ y = 0) ∨
    (x = -1 ∧ y = 0) ∨
    (x = 5 ∧ y = 2) ∨
    (x = -6 ∧ y = 2) :=
by sorry

end integer_solutions_of_equation_l3783_378364


namespace unique_a_value_l3783_378309

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, 
  (A a ∩ B).Nonempty ∧ 
  Set.Nonempty (A a ∩ B) ∧
  (A a ∩ C) = ∅ ∧
  a = -2 := by
  sorry

end unique_a_value_l3783_378309


namespace arithmetic_progression_sum_165_l3783_378379

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

theorem arithmetic_progression_sum_165 :
  ∃ ap : ArithmeticProgression,
    sum_n_terms ap 15 = 200 ∧
    sum_n_terms ap 150 = 150 ∧
    sum_n_terms ap 165 = -3064 := by
  sorry

end arithmetic_progression_sum_165_l3783_378379


namespace fixed_fee_december_l3783_378330

/-- Represents the billing information for an online service provider --/
structure BillingInfo where
  dec_fixed_fee : ℝ
  hourly_charge : ℝ
  dec_connect_time : ℝ
  jan_connect_time : ℝ
  dec_bill : ℝ
  jan_bill : ℝ
  jan_fee_increase : ℝ

/-- The fixed monthly fee in December is $10.80 --/
theorem fixed_fee_december (info : BillingInfo) : info.dec_fixed_fee = 10.80 :=
  by
  have h1 : info.dec_bill = 15.00 := by sorry
  have h2 : info.jan_bill = 25.40 := by sorry
  have h3 : info.jan_connect_time = 3 * info.dec_connect_time := by sorry
  have h4 : info.jan_fee_increase = 2 := by sorry
  have h5 : info.dec_fixed_fee + info.hourly_charge * info.dec_connect_time = info.dec_bill := by sorry
  have h6 : (info.dec_fixed_fee + info.jan_fee_increase) + info.hourly_charge * info.jan_connect_time = info.jan_bill := by sorry
  sorry

#check fixed_fee_december

end fixed_fee_december_l3783_378330


namespace vertex_y_coordinate_is_zero_l3783_378385

-- Define a trinomial function
def trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition that (f(x))^3 - f(x) = 0 has three real roots
def has_three_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (f x₁)^3 - f x₁ = 0 ∧ (f x₂)^3 - f x₂ = 0 ∧ (f x₃)^3 - f x₃ = 0

-- Theorem statement
theorem vertex_y_coordinate_is_zero 
  (a b c : ℝ) 
  (h : has_three_real_roots (trinomial a b c)) :
  let f := trinomial a b c
  let vertex_y := f (- b / (2 * a))
  vertex_y = 0 := by
sorry

end vertex_y_coordinate_is_zero_l3783_378385


namespace symmetric_line_l3783_378359

/-- Given a line L1 with equation 2x-y+3=0 and a point M(-1,2),
    prove that the line L2 symmetric to L1 with respect to M
    has the equation 2x-y+5=0 -/
theorem symmetric_line (x y : ℝ) :
  (2 * x - y + 3 = 0) →
  (2 * (-2 - x) - (4 - y) + 3 = 0) →
  (2 * x - y + 5 = 0) :=
by sorry

end symmetric_line_l3783_378359


namespace sqrt_inequality_l3783_378396

def M : Set ℝ := {x | 1 < x ∧ x < 4}

theorem sqrt_inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  |Real.sqrt (a * b) - 2| < |2 * Real.sqrt a - Real.sqrt b| := by
  sorry

end sqrt_inequality_l3783_378396


namespace triangle_area_l3783_378370

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3 under the following conditions:
    - (2b - √3c) / (√3a) = cos(C) / cos(A)
    - B = π/6
    - The median AM on side BC has length √7 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (M : ℝ) : 
  (2 * b - Real.sqrt 3 * c) / (Real.sqrt 3 * a) = Real.cos C / Real.cos A →
  B = π / 6 →
  M = Real.sqrt 7 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry


end triangle_area_l3783_378370


namespace sunset_time_correct_l3783_378350

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem sunset_time_correct 
  (sunrise : Time)
  (daylight : Duration)
  (sunset : Time)
  (h1 : sunrise = { hours := 6, minutes := 45 })
  (h2 : daylight = { hours := 11, minutes := 12 })
  (h3 : sunset = { hours := 17, minutes := 57 }) :
  addDuration sunrise daylight = sunset :=
sorry

end sunset_time_correct_l3783_378350


namespace f_is_quadratic_l3783_378313

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l3783_378313


namespace unique_solution_inequality_l3783_378333

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 5*a| ≤ 3) ↔ (a = 3/4 ∨ a = -3/4) := by
  sorry

end unique_solution_inequality_l3783_378333


namespace remainder_of_3_pow_19_l3783_378315

theorem remainder_of_3_pow_19 : 3^19 % 1162261460 = 7 := by
  sorry

end remainder_of_3_pow_19_l3783_378315


namespace choose_two_from_four_l3783_378335

theorem choose_two_from_four : Nat.choose 4 2 = 6 := by
  sorry

end choose_two_from_four_l3783_378335


namespace age_puzzle_solution_l3783_378319

/-- Represents a person in the age puzzle -/
structure Person where
  name : String
  age : Nat

/-- The conditions of the age puzzle -/
def AgePuzzle (tamara lena marina : Person) : Prop :=
  tamara.age = lena.age - 2 ∧
  tamara.age = marina.age + 1 ∧
  lena.age = marina.age + 3 ∧
  marina.age < tamara.age

/-- The theorem stating the unique solution to the age puzzle -/
theorem age_puzzle_solution :
  ∃! (tamara lena marina : Person),
    tamara.name = "Tamara" ∧
    lena.name = "Lena" ∧
    marina.name = "Marina" ∧
    AgePuzzle tamara lena marina ∧
    tamara.age = 23 ∧
    lena.age = 25 ∧
    marina.age = 22 := by
  sorry

end age_puzzle_solution_l3783_378319


namespace problem_statement_l3783_378392

theorem problem_statement : (12 : ℕ)^3 * 6^2 / 432 = 144 := by
  sorry

end problem_statement_l3783_378392


namespace equilibrium_instability_l3783_378320

/-- The system of differential equations -/
def system (x y : ℝ) : ℝ × ℝ :=
  (y^3 + x^5, x^3 + y^5)

/-- The Lyapunov function -/
def v (x y : ℝ) : ℝ :=
  x^4 - y^4

/-- The time derivative of the Lyapunov function -/
def dv_dt (x y : ℝ) : ℝ :=
  4 * (x^8 - y^8)

/-- Theorem stating the instability of the equilibrium point (0, 0) -/
theorem equilibrium_instability :
  ∃ (ε : ℝ), ε > 0 ∧
  ∀ (δ : ℝ), δ > 0 →
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 < δ^2 ∧
  ∃ (t : ℝ), t > 0 ∧
  let (x, y) := system x₀ y₀
  x^2 + y^2 > ε^2 :=
sorry

end equilibrium_instability_l3783_378320


namespace unique_real_sqrt_negative_square_l3783_378371

theorem unique_real_sqrt_negative_square : 
  ∃! x : ℝ, ∃ y : ℝ, y ^ 2 = -(x + 2) ^ 2 := by sorry

end unique_real_sqrt_negative_square_l3783_378371


namespace binary_digits_difference_l3783_378303

theorem binary_digits_difference : ∃ n m : ℕ, 
  (2^n ≤ 300 ∧ 300 < 2^(n+1)) ∧ 
  (2^m ≤ 1400 ∧ 1400 < 2^(m+1)) ∧ 
  m - n = 2 := by
sorry

end binary_digits_difference_l3783_378303


namespace polygon_sides_from_interior_angle_l3783_378329

theorem polygon_sides_from_interior_angle (n : ℕ) (angle : ℝ) : 
  (n ≥ 3) → (angle = 140) → (n * angle = (n - 2) * 180) → n = 9 := by
  sorry

end polygon_sides_from_interior_angle_l3783_378329


namespace other_pencil_length_l3783_378317

/-- Given two pencils with a total length of 24 cubes, where one pencil is 12 cubes long,
    prove that the other pencil is also 12 cubes long. -/
theorem other_pencil_length (total_length : ℕ) (first_pencil : ℕ) (h1 : total_length = 24) (h2 : first_pencil = 12) :
  total_length - first_pencil = 12 := by
  sorry

end other_pencil_length_l3783_378317


namespace different_tens_digit_probability_l3783_378383

theorem different_tens_digit_probability :
  let n : ℕ := 5  -- number of integers to choose
  let lower_bound : ℕ := 10  -- lower bound of the range
  let upper_bound : ℕ := 59  -- upper bound of the range
  let total_numbers : ℕ := upper_bound - lower_bound + 1  -- total numbers in the range
  let tens_digits : ℕ := 5  -- number of different tens digits in the range
  let numbers_per_tens : ℕ := 10  -- numbers available for each tens digit

  -- Probability of choosing n integers with different tens digits
  (numbers_per_tens ^ n : ℚ) / (total_numbers.choose n) = 2500 / 52969 :=
by sorry

end different_tens_digit_probability_l3783_378383


namespace rabbits_ate_27_watermelons_l3783_378387

/-- The number of watermelons eaten by rabbits, given initial and remaining counts. -/
def watermelons_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem stating that 27 watermelons were eaten by rabbits. -/
theorem rabbits_ate_27_watermelons : 
  watermelons_eaten 35 8 = 27 := by sorry

end rabbits_ate_27_watermelons_l3783_378387


namespace perpendicular_lines_a_value_l3783_378300

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ x y : ℝ, a^2 * x + y + 7 = 0 ∧ x - 2 * a * y + 1 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (a^2 * x₁ + y₁ + 7 = 0 ∧ x₁ - 2 * a * y₁ + 1 = 0) →
    (a^2 * x₂ + y₂ + 7 = 0 ∧ x₂ - 2 * a * y₂ + 1 = 0) →
    (x₂ - x₁) * (a^2 * (x₂ - x₁) + (y₂ - y₁)) = 0) →
  a = 0 ∨ a = 2 :=
by sorry

end perpendicular_lines_a_value_l3783_378300


namespace concentric_circles_equal_areas_l3783_378380

/-- Given a circle of radius R divided by two concentric circles into three equal areas,
    prove that the radii of the concentric circles are R/√3 and R√(2/3) -/
theorem concentric_circles_equal_areas (R : ℝ) (R₁ R₂ : ℝ) (h₁ : R > 0) :
  (π * R₁^2 = π * R^2 / 3) ∧ 
  (π * R₂^2 - π * R₁^2 = π * R^2 / 3) ∧ 
  (π * R^2 - π * R₂^2 = π * R^2 / 3) →
  (R₁ = R / Real.sqrt 3) ∧ (R₂ = R * Real.sqrt (2/3)) := by
  sorry

end concentric_circles_equal_areas_l3783_378380


namespace binomial_sum_divides_power_of_two_l3783_378357

theorem binomial_sum_divides_power_of_two (n : ℕ) : 
  n > 3 → (1 + n.choose 1 + n.choose 2 + n.choose 3 ∣ 2^2000) ↔ (n = 7 ∨ n = 23) := by
  sorry

end binomial_sum_divides_power_of_two_l3783_378357


namespace exam_score_distribution_l3783_378337

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  mean : ℝ
  stdDev : ℝ
  totalStudents : ℕ

/-- Calculates the number of students who scored at least a given threshold -/
def studentsAboveThreshold (dist : ScoreDistribution) (threshold : ℝ) : ℕ :=
  sorry

/-- The exam score distribution -/
def examScores : ScoreDistribution :=
  { mean := 110
    stdDev := 10
    totalStudents := 50 }

theorem exam_score_distribution :
  (studentsAboveThreshold examScores 90 = 49) ∧
  (studentsAboveThreshold examScores 120 = 8) := by
  sorry

end exam_score_distribution_l3783_378337


namespace original_number_proof_l3783_378390

theorem original_number_proof (x : ℝ) : 
  (x * 1.2 * 0.6 = 1080) → x = 1500 := by
  sorry

end original_number_proof_l3783_378390


namespace nathaniel_best_friends_l3783_378347

def initial_tickets : ℕ := 11
def remaining_tickets : ℕ := 3
def tickets_per_friend : ℕ := 2

def number_of_friends : ℕ := (initial_tickets - remaining_tickets) / tickets_per_friend

theorem nathaniel_best_friends : number_of_friends = 4 := by
  sorry

end nathaniel_best_friends_l3783_378347


namespace horner_method_correct_f_3_equals_283_l3783_378384

def f (x : ℝ) : ℝ := x^5 + x^3 + x^2 + x + 1

def horner_eval (x : ℝ) : ℝ := ((((1 * x + 0) * x + 1) * x + 1) * x + 1) * x + 1

theorem horner_method_correct (x : ℝ) : f x = horner_eval x := by sorry

theorem f_3_equals_283 : f 3 = 283 := by sorry

end horner_method_correct_f_3_equals_283_l3783_378384


namespace original_denominator_proof_l3783_378341

theorem original_denominator_proof (d : ℚ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3) / (d + 3) = 2 / 3 →
  d = 15 / 2 := by
sorry

end original_denominator_proof_l3783_378341
