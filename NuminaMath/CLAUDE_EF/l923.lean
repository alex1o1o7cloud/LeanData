import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tournament_result_l923_92383

-- Define the teams
inductive Team
  | A | B | C | D | E
deriving Repr, DecidableEq

-- Define the possible match outcomes
inductive Outcome
  | Win | Loss | Draw
deriving Repr, DecidableEq

-- Define the tournament results type
def TournamentResults := Team → Team → Outcome

-- Define the point system
def points (o : Outcome) : Nat :=
  match o with
  | Outcome.Win => 2
  | Outcome.Draw => 1
  | Outcome.Loss => 0

-- Calculate total points for a team
def totalPoints (results : TournamentResults) (t : Team) : Nat :=
  List.sum (List.map (fun opponent => 
    if t = opponent then 0
    else points (results t opponent)) [Team.A, Team.B, Team.C, Team.D, Team.E])

-- Define the tournament conditions
def validTournament (results : TournamentResults) : Prop :=
  -- Team A did not have any draws
  (∀ t, results Team.A t ≠ Outcome.Draw ∧ results t Team.A ≠ Outcome.Draw) ∧
  -- Team B did not lose any matches
  (∀ t, results Team.B t ≠ Outcome.Loss) ∧
  -- Team D did not win any matches
  (∀ t, results Team.D t ≠ Outcome.Win) ∧
  -- All teams scored a different number of points
  (∀ t1 t2, t1 ≠ t2 → totalPoints results t1 ≠ totalPoints results t2) ∧
  -- Teams finished in the order A, B, C, D, E
  (totalPoints results Team.A > totalPoints results Team.B) ∧
  (totalPoints results Team.B > totalPoints results Team.C) ∧
  (totalPoints results Team.C > totalPoints results Team.D) ∧
  (totalPoints results Team.D > totalPoints results Team.E)

-- Define the expected results
def expectedResults : TournamentResults := fun t1 t2 =>
  match t1, t2 with
  | Team.A, Team.B => Outcome.Loss
  | Team.A, _ => Outcome.Win
  | Team.B, Team.A => Outcome.Win
  | Team.B, _ => Outcome.Draw
  | Team.C, Team.A => Outcome.Loss
  | Team.C, _ => Outcome.Draw
  | Team.D, Team.A => Outcome.Loss
  | Team.D, Team.E => Outcome.Win
  | Team.D, _ => Outcome.Draw
  | Team.E, Team.A => Outcome.Loss
  | Team.E, Team.D => Outcome.Loss
  | Team.E, _ => Outcome.Draw

-- Theorem statement
theorem unique_tournament_result :
  ∀ results : TournamentResults,
    validTournament results →
    results = expectedResults := by
  sorry

#eval totalPoints expectedResults Team.A
#eval totalPoints expectedResults Team.B
#eval totalPoints expectedResults Team.C
#eval totalPoints expectedResults Team.D
#eval totalPoints expectedResults Team.E

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tournament_result_l923_92383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graphene_thickness_scientific_notation_l923_92366

def graphene_thickness : ℝ := 0.00000000034

theorem graphene_thickness_scientific_notation :
  graphene_thickness = 3.4 * (10 : ℝ)^(-10 : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graphene_thickness_scientific_notation_l923_92366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l923_92394

/-- Calculates the length of a tunnel given train specifications and time to pass through --/
theorem tunnel_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_minutes : ℝ)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 72)
  (h3 : time_minutes = 2) :
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let time_seconds := time_minutes * 60
  let distance_traveled := train_speed_ms * time_seconds
  let tunnel_length := distance_traveled - train_length
  tunnel_length = 2300 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l923_92394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_half_angle_l923_92345

theorem sin_plus_cos_half_angle (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : 2 * Real.pi < α) 
  (h3 : α < 3 * Real.pi) : 
  Real.sin (α/2) + Real.cos (α/2) = -(2 * Real.sqrt 3)/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_half_angle_l923_92345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_eggs_count_l923_92380

theorem goose_eggs_count (total_eggs : ℕ) : 
  (↑total_eggs * (2 : ℚ) / 3 : ℚ).num ≥ 0 →
  (↑total_eggs * (2 : ℚ) / 3 * (3 : ℚ) / 4 : ℚ).num ≥ 0 →
  ↑total_eggs * (2 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 = 100 →
  total_eggs = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_eggs_count_l923_92380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l923_92372

theorem equation_solution :
  ∀ x : ℝ, (2 : ℝ) ^ ((16 : ℝ) ^ (x^2)) = (16 : ℝ) ^ ((2 : ℝ) ^ (x^2)) ↔ 
    x = Real.sqrt (2/3) ∨ x = -Real.sqrt (2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l923_92372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rearrangement_with_different_sums_l923_92326

/-- Represents a 2 × n table of real numbers -/
def Table (n : ℕ) := Fin 2 → Fin n → ℝ

/-- The sum of a column in the table -/
def columnSum (t : Table n) (j : Fin n) : ℝ :=
  (t 0 j) + (t 1 j)

/-- The sum of a row in the table -/
def rowSum (t : Table n) (i : Fin 2) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin n)) (λ j => t i j)

/-- Predicate for all column sums being different -/
def allColumnSumsDifferent (t : Table n) : Prop :=
  ∀ j₁ j₂, j₁ ≠ j₂ → columnSum t j₁ ≠ columnSum t j₂

/-- Predicate for all row sums being different -/
def allRowSumsDifferent (t : Table n) : Prop :=
  rowSum t 0 ≠ rowSum t 1

/-- Main theorem: For any 2 × n table (n > 2) with different column sums,
    there exists a rearrangement with both different column and row sums -/
theorem exists_rearrangement_with_different_sums (n : ℕ) (h : n > 2) :
  ∀ t : Table n, allColumnSumsDifferent t →
  ∃ t' : Table n, (∀ j, columnSum t j = columnSum t' j) ∧
                   allColumnSumsDifferent t' ∧
                   allRowSumsDifferent t' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rearrangement_with_different_sums_l923_92326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_complete_job_in_six_days_l923_92334

/-- The time (in days) it takes for two workers to complete a job together,
    given the time it takes for each worker to complete the job individually. -/
noncomputable def time_together (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem stating that if two workers can each complete a job in 12 days individually,
    then together they can complete the job in 6 days. -/
theorem workers_complete_job_in_six_days (time_a time_b : ℝ) 
  (ha : time_a = 12) (hb : time_b = 12) : 
  time_together time_a time_b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_complete_job_in_six_days_l923_92334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_smaller_base_l923_92303

-- Define the trapezoid
structure Trapezoid where
  p : ℝ
  q : ℝ
  a : ℝ
  h : 0 < p
  h' : p < q
  h'' : 0 < a

-- Define the condition for angles at the larger base
def angleRatio (t : Trapezoid) : Prop :=
  ∃ α : ℝ, 0 < α ∧ α < Real.pi/2 ∧
    t.p * Real.sin (2*α) = t.q * Real.sin α

-- Theorem statement
theorem trapezoid_smaller_base (t : Trapezoid) 
  (angle_cond : angleRatio t) :
  ∃ b : ℝ, b = (t.p^2 + t.a*t.p - t.q^2) / t.p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_smaller_base_l923_92303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_example_harmonious_log_range_l923_92338

-- Define the concept of harmonious function
def is_harmonious (f₁ f₂ h : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, h x = a * f₁ x + b * f₂ x

-- Part 1
theorem harmonious_example :
  is_harmonious (λ x => x - 1) (λ x => 3 * x + 1) (λ x => 2 * x + 2) :=
by sorry

-- Part 2
theorem harmonious_log_range :
  let f₁ : ℝ → ℝ := λ x => Real.log x / Real.log 3
  let f₂ : ℝ → ℝ := λ x => Real.log x / Real.log (1/3)
  let h : ℝ → ℝ := λ x => 2 * f₁ x + f₂ x
  ∀ t : ℝ, (∃ x ∈ Set.Icc 3 9, h (9*x) + t * h (3*x) = 0) ↔ t ∈ Set.Icc (-3/2) (-4/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_example_harmonious_log_range_l923_92338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l923_92331

/-- The function f(x) = 2x + ln x - 6 --/
noncomputable def f (x : ℝ) : ℝ := 2*x + Real.log x - 6

/-- m is a root of f --/
def is_root (m : ℝ) : Prop := f m = 0

theorem root_in_interval :
  ∃ m : ℝ, is_root m ∧ 2 < m ∧ m < 3 := by
  sorry

#check root_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l923_92331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banach_matches_problem_l923_92353

/-- The expected number of matches remaining in one box when the other becomes empty -/
noncomputable def expected_matches (n : ℕ) : ℝ :=
  2 * Real.sqrt ((n + 1 : ℝ) / Real.pi) - 1

/-- Theorem stating the expected number of matches for the given problem -/
theorem banach_matches_problem (n : ℕ) (ε : ℝ) (h_ε : ε > 0) :
  ∃ (N : ℕ), ∀ (m : ℕ), m ≥ N → 
    |expected_matches m - (2 * Real.sqrt ((m + 1 : ℝ) / Real.pi) - 1)| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banach_matches_problem_l923_92353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_zero_on_positive_x_axis_l923_92367

theorem sine_tangent_zero_on_positive_x_axis (α : ℝ) :
  (∃ k : ℤ, α = 2 * Real.pi * k) → Real.sin α = 0 ∧ Real.tan α = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_zero_on_positive_x_axis_l923_92367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_nest_problem_l923_92389

/-- Given a bird building a nest, calculate the number of twigs it still needs to find. -/
def bird_nest_twigs (base_twigs : ℕ) (additional_per_twig : ℕ) (tree_fraction : ℚ) : ℕ :=
  let total_additional := base_twigs * additional_per_twig
  let tree_dropped := (total_additional : ℚ) * tree_fraction
  total_additional - Int.toNat tree_dropped.floor

/-- The bird will need to find 48 more twigs to finish its nest. -/
theorem bird_nest_problem : bird_nest_twigs 12 6 (1/3) = 48 := by
  -- Unfold the definition of bird_nest_twigs
  unfold bird_nest_twigs
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl

#eval bird_nest_twigs 12 6 (1/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_nest_problem_l923_92389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_equation_solution_l923_92316

-- Define the Euler's equation
def euler_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x^2 * ((deriv (deriv y)) x) - x * ((deriv y) x) + 2 * (y x) = x * (Real.log x)

-- Define the proposed solution
noncomputable def proposed_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  x * (C₁ * Real.cos (Real.log x) + C₂ * Real.sin (Real.log x)) + x * Real.log x

-- Theorem statement
theorem euler_equation_solution (C₁ C₂ : ℝ) :
  ∀ x > 0, euler_equation (proposed_solution C₁ C₂) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_equation_solution_l923_92316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_cost_is_474_l923_92329

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of walls in a room -/
def wallSurfaceArea (room : Dimensions) : ℝ :=
  2 * (room.length * room.height + room.width * room.height)

/-- Calculates the area of a rectangular opening -/
def openingArea (d : Dimensions) : ℝ :=
  d.length * d.width

/-- Represents the room and its openings -/
structure Room where
  dimensions : Dimensions
  doorDimensions : Dimensions
  numDoors : ℕ
  largeWindowDimensions : Dimensions
  numLargeWindows : ℕ
  smallWindowDimensions : Dimensions
  numSmallWindows : ℕ

/-- Calculates the total area of openings in the room -/
def totalOpeningArea (r : Room) : ℝ :=
  r.numDoors * openingArea r.doorDimensions +
  r.numLargeWindows * openingArea r.largeWindowDimensions +
  r.numSmallWindows * openingArea r.smallWindowDimensions

/-- Calculates the paintable area in the room -/
def paintableArea (r : Room) : ℝ :=
  wallSurfaceArea r.dimensions - totalOpeningArea r

/-- Theorem: The cost of painting the walls is Rs. 474 -/
theorem painting_cost_is_474 : ∃ (room : Room) (costPerSqM : ℝ), paintableArea room * costPerSqM = 474 := by
  let room : Room := {
    dimensions := { length := 10, width := 7, height := 5 }
    doorDimensions := { length := 1, width := 3, height := 0 }
    numDoors := 2
    largeWindowDimensions := { length := 2, width := 1.5, height := 0 }
    numLargeWindows := 1
    smallWindowDimensions := { length := 1, width := 1.5, height := 0 }
    numSmallWindows := 2
  }
  let costPerSqM : ℝ := 3
  exists room, costPerSqM
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_cost_is_474_l923_92329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_helicopter_performance_l923_92356

def height_changes : List ℝ := [4.1, -2.3, 1.6, -0.9, 1.1]
def ascent_fuel_rate : ℝ := 5
def descent_fuel_rate : ℝ := 3

theorem helicopter_performance :
  (height_changes.sum = 3.6) ∧
  ((height_changes.map (fun h => if h > 0 then h * ascent_fuel_rate else -h * descent_fuel_rate)).sum = 43.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_helicopter_performance_l923_92356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l923_92371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 5) * x - 2
  else x^2 - 2 * (a + 1) * x + 3 * a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc 1 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l923_92371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_book_price_is_eleven_l923_92391

def greatest_book_price (total_budget : ℕ) (num_books : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  let remaining_budget := total_budget - entrance_fee
  let max_price_with_tax := (remaining_budget : ℚ) / num_books
  ⌊(max_price_with_tax / (1 + tax_rate))⌋.toNat

theorem max_book_price_is_eleven :
  greatest_book_price 250 20 5 (7/100) = 11 := by
  rfl

#eval greatest_book_price 250 20 5 (7/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_book_price_is_eleven_l923_92391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_cos_value_l923_92392

-- Define α as a real number
variable (α : ℝ)

-- Define the condition that tan α = 2
def tan_alpha : ℝ → Prop := λ α => Real.tan α = 2

-- Define the condition that α is in the third quadrant
def third_quadrant : ℝ → Prop := λ α => Real.cos α < 0 ∧ Real.sin α < 0

-- Theorem 1
theorem fraction_value (h : tan_alpha α) :
  (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 := by
  sorry

-- Theorem 2
theorem cos_value (h1 : tan_alpha α) (h2 : third_quadrant α) :
  Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_cos_value_l923_92392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_count_l923_92384

/-- Represents the number of distinct arrangements of plants under lamps -/
def distinct_arrangements : ℕ := 5

/-- Number of ferns -/
def num_ferns : ℕ := 3

/-- Number of rubber plants -/
def num_rubber_plants : ℕ := 1

/-- Number of blue lamps -/
def num_blue_lamps : ℕ := 3

/-- Number of yellow lamps -/
def num_yellow_lamps : ℕ := 2

/-- Total number of plants -/
def total_plants : ℕ := num_ferns + num_rubber_plants

/-- Total number of lamps -/
def total_lamps : ℕ := num_blue_lamps + num_yellow_lamps

/-- Represents whether a plant is under a lamp in a given arrangement -/
def plant_under_lamp (arrangement : ℕ) (plant : ℕ) (lamp : ℕ) : Prop := sorry

theorem plant_arrangement_count :
  distinct_arrangements = 5 ∧
  num_ferns = 3 ∧
  num_rubber_plants = 1 ∧
  num_blue_lamps = 3 ∧
  num_yellow_lamps = 2 ∧
  total_plants ≤ total_lamps ∧
  (∀ arrangement, arrangement ≤ distinct_arrangements →
    (∀ plant, plant ≤ total_plants → ∃! lamp, lamp ≤ total_lamps ∧ plant_under_lamp arrangement plant lamp)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plant_arrangement_count_l923_92384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l923_92314

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where A is an endpoint of the minor axis and F₁, F₂ are the foci. -/
def Ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b

/-- The cosine of angle F₁AF₂ is 5/6 -/
def CosineCondition (a c : ℝ) : Prop :=
  (5 : ℝ) / 6 = (2 * a^2 - 4 * c^2) / (2 * a^2)

/-- The eccentricity of an ellipse -/
noncomputable def Eccentricity (c a : ℝ) : ℝ := c / a

/-- Theorem: If the cosine condition holds for an ellipse, 
    then its eccentricity is √3/6 -/
theorem ellipse_eccentricity 
  (a b c : ℝ) 
  (h_ellipse : Ellipse a b) 
  (h_cosine : CosineCondition a c) : 
  Eccentricity c a = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l923_92314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_sum_l923_92369

theorem gcd_power_sum (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) (h3 : Nat.Coprime m n) :
  Nat.gcd (5^m + 7^m) (5^n + 7^n) = if Even (m + n) then 12 else 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_sum_l923_92369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_cash_adjustment_l923_92390

/-- Represents the net error in cents due to miscounting coins in a coffee shop. -/
def net_error (x y z : ℕ) : ℤ :=
  25 * y + 50 * x - 4 * z

/-- Proves that subtracting the net error gives the correct cash amount. -/
theorem correct_cash_adjustment (x y z : ℕ) (total : ℤ) :
  total - net_error x y z = total - (25 * y + 50 * x - 4 * z) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_cash_adjustment_l923_92390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l923_92397

theorem polynomial_root_sum (a b c d e : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x ∈ ({5, -3, 2} : Set ℝ)) →
  (b + d) / a = -2677 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_sum_l923_92397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l923_92308

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 16 * Real.sqrt 2) :
  let r : ℝ := d / (4 * Real.sqrt 2)
  (4 / 3) * Real.pi * r^3 = 2048 / 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l923_92308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l923_92398

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∀ y, 0 < y ∧ y < Real.pi / 2 → f (Real.pi / 6 - y) = f (Real.pi / 6 + y)) →
  x = Real.pi / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l923_92398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l923_92350

/-- Parabola with vertex at origin, symmetrical about coordinate axis, and directrix x = -1 -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 4 * p.1}

/-- Ellipse 3x^2 + 2y^2 = 2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | 3 * p.1^2 + 2 * p.2^2 = 2}

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus with slope tan(α) -/
def Line (α : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = Real.tan α * (p.1 - 1)}

/-- Chord length |AB| for a line intersecting the parabola -/
noncomputable def chordLength (α : ℝ) : ℝ :=
  let k := Real.tan α
  (2 * k^2 + 4) / k^2 + 2

/-- Theorem: If the chord length is ≤ 8 and the line intersects the ellipse,
    then the slope angle α is in the specified range -/
theorem parabola_line_intersection
  (α : ℝ)
  (chord_condition : chordLength α ≤ 8)
  (ellipse_intersection : (Line α ∩ Ellipse).Nonempty)
  : π/4 ≤ α ∧ α ≤ π/3 ∨ 2*π/3 ≤ α ∧ α ≤ 3*π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l923_92350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l923_92327

theorem matrix_satisfies_conditions : ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M.mulVec (![4, 0] : Fin 2 → ℝ) = ![8, 28] ∧
  M.mulVec (![(-2), 6] : Fin 2 → ℝ) = ![2, (-20)] :=
by
  -- The matrix M that satisfies the conditions
  let M : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 1], ![7, -1]]
  
  -- Assert the existence of M
  use M

  -- Proof that M satisfies the conditions
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l923_92327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_at_one_l923_92386

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem instantaneous_rate_of_change_at_one :
  (deriv f) 1 = -1 := by
  -- We'll use the derivative of 1/x and evaluate it at x = 1
  have h1 : deriv f = fun x => -(1 / x^2) := by
    -- This is where we would prove that the derivative of 1/x is -1/x^2
    sorry
  -- Now we evaluate this at x = 1
  calc
    (deriv f) 1 = (fun x => -(1 / x^2)) 1 := by rw [h1]
    _ = -(1 / 1^2) := by rfl
    _ = -1 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_rate_of_change_at_one_l923_92386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_points_properties_l923_92344

/-- Given a triangle OAB and points P and Q, prove properties about their coordinates and dot products. -/
theorem triangle_points_properties (O A B P Q : ℝ × ℝ) (l : ℝ) :
  O = (0, 0) →
  A = (2, 9) →
  B = (6, -3) →
  P.1 = 14 →
  P - O = l • (B - P) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • B →
  (P - O) • (P - A) = 0 →
  l = -7/4 ∧
  P = (14, -7) ∧
  Q = (4, 3) ∧
  ∀ R : ℝ × ℝ, (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ R = (1 - s) • O + s • Q) →
    -25/2 ≤ (O - R) • ((A - R) + (B - R)) ∧ (O - R) • ((A - R) + (B - R)) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_points_properties_l923_92344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ski_resort_snowfall_l923_92354

/-- Calculates the average snowfall per hour given total snowfall and time period. -/
noncomputable def average_snowfall_per_hour (total_snowfall : ℝ) (time_period_weeks : ℝ) : ℝ :=
  total_snowfall / (time_period_weeks * 7 * 24)

/-- Proves that the average snowfall per hour is 5/4 inches given the conditions. -/
theorem ski_resort_snowfall :
  average_snowfall_per_hour 210 1 = 5 / 4 := by
  -- Unfold the definition of average_snowfall_per_hour
  unfold average_snowfall_per_hour
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check ski_resort_snowfall

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ski_resort_snowfall_l923_92354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_theorem_l923_92337

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 3)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem tan_domain_theorem :
  domain f = {x : ℝ | ∀ k : ℤ, x ≠ 5 * Real.pi / 12 + k * Real.pi / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_domain_theorem_l923_92337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l923_92343

/-- A coloring function for lattice points -/
def ColoringFunction := (ℤ × ℤ) → Fin 10

/-- The set of valid lattice points -/
def ValidLatticePoints : Set (ℤ × ℤ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ 252 ∧ 1 ≤ p.2 ∧ p.2 ≤ 252 ∧ p.1 ≠ p.2}

/-- The coloring satisfies the distinctness property -/
def SatisfiesDistinctness (f : ColoringFunction) : Prop :=
  ∀ a b c : ℤ, 1 ≤ a ∧ a ≤ 252 → 1 ≤ b ∧ b ≤ 252 → 1 ≤ c ∧ c ≤ 252 →
    a ≠ b → b ≠ c → f (a, b) ≠ f (b, c)

/-- There exists a valid coloring function -/
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (∀ p ∈ ValidLatticePoints, f p ∈ Finset.univ) ∧
  SatisfiesDistinctness f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l923_92343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l923_92309

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if two lines are parallel -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- The asymptote of a hyperbola -/
def asymptote (h : Hyperbola) : Line :=
  { a := h.b, b := -h.a, c := 0 }

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a^2 + h.b^2) / h.b^2)

/-- Theorem stating the eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : parallel (asymptote h) { a := 1, b := -1, c := 1 }) :
  eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l923_92309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alan_cd_purchase_l923_92357

/-- The price of a CD by "The Dark" -/
def dark_cd_price : ℚ := 24

/-- The price of a CD by "AVN" -/
def avn_cd_price : ℚ := 12

/-- The number of CDs by "The Dark" Alan buys -/
def dark_cd_count : ℕ := 2

/-- The number of CDs by "AVN" Alan buys -/
def avn_cd_count : ℕ := 1

/-- The number of 90s mix CDs Alan buys -/
def mix_cd_count : ℕ := 5

/-- The cost of 90s mix CDs as a percentage of other CDs -/
def mix_cd_cost_percentage : ℚ := 2/5

/-- The discount percentage on the total purchase -/
def discount_percentage : ℚ := 1/10

/-- The sales tax percentage -/
def sales_tax_percentage : ℚ := 2/25

/-- The final amount Alan has to pay, rounded to the nearest cent -/
def final_payment : ℚ := 8165/100

theorem alan_cd_purchase :
  let other_cds_cost := dark_cd_price * dark_cd_count + avn_cd_price * avn_cd_count
  let mix_cds_cost := mix_cd_cost_percentage * other_cds_cost
  let total_before_discount := other_cds_cost + mix_cds_cost
  let discount_amount := discount_percentage * total_before_discount
  let discounted_total := total_before_discount - discount_amount
  let sales_tax := sales_tax_percentage * discounted_total
  let total_with_tax := discounted_total + sales_tax
  (Int.floor (total_with_tax * 100 + 1/2) / 100) = final_payment := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alan_cd_purchase_l923_92357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l923_92328

/-- Represents a cube with 8 vertices -/
structure Cube where
  vertices : Fin 8

/-- Represents a coloring of the cube -/
def Coloring := Cube → Fin 4

/-- Adjacency relation between vertices -/
def are_adjacent : Fin 8 → Fin 8 → Prop := sorry

/-- Checks if a coloring is valid (adjacent vertices have different colors) -/
def is_valid_coloring (c : Coloring) : Prop := 
  ∀ v1 v2 : Fin 8, are_adjacent v1 v2 → c ⟨v1⟩ ≠ c ⟨v2⟩

/-- The number of valid colorings of the cube -/
noncomputable def num_valid_colorings : ℕ := sorry

theorem cube_coloring_count : num_valid_colorings = 2652 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l923_92328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_equals_one_l923_92315

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 1 / 2019
  | n + 1 => a n + 1 / ((n + 1) * (n + 2))

theorem a_2019_equals_one :
  a 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_equals_one_l923_92315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_and_integer_constraint_l923_92333

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / (2 * x)

theorem tangent_lines_and_integer_constraint (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (f a x₁ - 0 = (deriv (f a)) x₁ * (x₁ - 1)) ∧
    (f a x₂ - 0 = (deriv (f a)) x₂ * (x₂ - 1)) ∧
    (∃! n : ℤ, (↑n : ℝ) ∈ Set.Ioo x₁ x₂)) →
  a ∈ Set.Icc (- 8 / 3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_and_integer_constraint_l923_92333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l923_92349

/-- The curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ := 
  Real.sqrt (4 / (1 + 3 * Real.cos θ ^ 2))

/-- The sum of squares of distances OA and OB -/
noncomputable def sum_of_squares (θ : ℝ) : ℝ :=
  (curve_C θ) ^ 2 + (curve_C (θ + Real.pi / 2)) ^ 2

/-- Theorem stating the minimum value of |OA|² + |OB|² -/
theorem min_sum_of_squares :
  ∃ (min_val : ℝ), min_val = 16/5 ∧ 
  ∀ θ, sum_of_squares θ ≥ min_val := by
  sorry

#check min_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l923_92349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_gt_f_one_l923_92306

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the property of being even on [-5, 5]
def isEvenOn (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (-5) 5 → f (-x) = f x

-- Define the property of being monotonic on [0, 5]
def isMonotonicOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 0 5 → y ∈ Set.Icc 0 5 → x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- State the theorem
theorem f_zero_gt_f_one
    (h_even : isEvenOn f)
    (h_monotonic : isMonotonicOn f)
    (h_inequality : f (-3) < f 1) :
    f 0 > f 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_gt_f_one_l923_92306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_returns_to_original_position_l923_92301

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the transformation
def transform (q : Quadrilateral) : Quadrilateral :=
  sorry

-- Define the condition for original position
def is_original_position (q q' : Quadrilateral) : Prop :=
  q.A = q'.A ∧ q.D = q'.D

-- Helper function to iterate the transformation
def iterate (f : α → α) : ℕ → α → α
  | 0, x => x
  | n + 1, x => f (iterate f n x)

-- Main theorem
theorem quadrilateral_returns_to_original_position 
  (q : Quadrilateral)
  (h1 : dist q.A q.B = 1)
  (h2 : dist q.B q.C = 1)
  (h3 : dist q.C q.D = 1)
  (h4 : dist q.A q.D ≠ 1) :
  ∃ n : ℕ, is_original_position q (iterate transform n q) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_returns_to_original_position_l923_92301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_proof_l923_92347

theorem linear_equation_proof (k m : ℝ) : 
  (∀ x, (abs k - 3) * x^2 - (k - 3) * x + 2 * m + 1 = 0 → (abs k - 3 = 0 ∧ k - 3 ≠ 0)) →
  (∀ x, (abs k - 3) * x^2 - (k - 3) * x + 2 * m + 1 = 0 ↔ 3 * x - 2 = 4 - 5 * x + 2 * x) →
  k = -3 ∧ m = 5/2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_proof_l923_92347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percy_swim_days_l923_92335

/-- Represents the number of days Percy swims before and after school per week -/
def days_per_week : ℕ := sorry

/-- Total hours Percy swims over 4 weeks -/
def total_hours : ℕ := 52

/-- Hours Percy swims on weekends per week -/
def weekend_hours : ℕ := 3

/-- Number of weeks -/
def num_weeks : ℕ := 4

/-- Hours Percy swims before and after school each day -/
def school_day_hours : ℕ := 2

theorem percy_swim_days :
  days_per_week = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percy_swim_days_l923_92335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_binomial_proof_l923_92305

/-- A polynomial is quadratic if its degree is 2 -/
def IsQuadratic (p : Polynomial ℝ) : Prop :=
  p.degree = 2

/-- A polynomial is a binomial if it has exactly two terms -/
def IsBinomial (p : Polynomial ℝ) : Prop :=
  p.support.card = 2

/-- The polynomial 4x^2 - 3 -/
noncomputable def p : Polynomial ℝ := 4 * Polynomial.X^2 - 3

theorem quadratic_binomial_proof : IsQuadratic p ∧ IsBinomial p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_binomial_proof_l923_92305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_questions_count_l923_92360

-- Define the number of marks per question
def marks_per_question : ℕ := 2

-- Define the scores of Alisson, Jose, and Meghan
def alisson_score : ℕ → ℕ := λ a => a

def jose_score (a : ℕ) : ℕ := a + 40

def meghan_score (a : ℕ) : ℕ := a + 20

-- Define the number of questions Jose got wrong
def jose_wrong_questions : ℕ := 5

-- Define the total score for all three
def total_score : ℕ := 210

-- Theorem to prove
theorem test_questions_count : 
  ∃ (a : ℕ), 
    alisson_score a + jose_score a + meghan_score a = total_score ∧
    (jose_score a + jose_wrong_questions * marks_per_question) / marks_per_question = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_questions_count_l923_92360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_sum_divisibility_l923_92304

theorem permutation_sum_divisibility
  (n : ℕ)
  (h_odd : Odd n)
  (h_gt_one : n > 1)
  (k : Fin n → ℤ) :
  ∃ (b c : Equiv.Perm (Fin n)), b ≠ c ∧
    (n.factorial : ℤ) ∣ (Finset.sum (Finset.univ : Finset (Fin n)) (λ i => k i * b i) -
                         Finset.sum (Finset.univ : Finset (Fin n)) (λ i => k i * c i)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_sum_divisibility_l923_92304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_constant_chord_length_l923_92320

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the distance from a point to the focus
noncomputable def distToFocus (p x y : ℝ) : ℝ := ((x - p/2)^2 + y^2).sqrt

-- Define the line through two points
def lineThrough (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

-- Define perpendicularity of two lines through the origin
def perpThruOrigin (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

-- Define the circle
def circleEq (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 1

theorem parabola_intersection_ratio 
  (p : ℝ) (t : ℝ) (x_q y_q : ℝ) :
  parabola p 2 t →
  distToFocus p 2 t = 5/2 →
  lineThrough (-1/2) 0 2 t x_q y_q →
  parabola p x_q y_q →
  distToFocus p x_q y_q / distToFocus p 2 t = 1/4 := by sorry

theorem constant_chord_length 
  (p a : ℝ) (x_a y_a x_b y_b x_d y_d x_e y_e : ℝ) :
  parabola p x_a y_a →
  parabola p x_b y_b →
  circleEq a x_d y_d →
  circleEq a x_e y_e →
  perpThruOrigin x_a y_a x_b y_b →
  (∃ k, ∀ x_d y_d x_e y_e,
    circleEq a x_d y_d → circleEq a x_e y_e →
    (x_e - x_d)^2 + (y_e - y_d)^2 = k^2) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_constant_chord_length_l923_92320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_C₁_to_C₂_l923_92385

/-- Curve C₁ is a circle with center (-4, 3) and radius 2 --/
def C₁ (x y : ℝ) : Prop :=
  (x + 4)^2 + (y - 3)^2 = 4

/-- Curve C₂ is a straight line with equation x - y = 4 --/
def C₂ (x y : ℝ) : Prop :=
  x - y = 4

/-- The distance between two points in ℝ² --/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem stating that the point on C₁ closest to C₂ has coordinates (-4+√2, 3-√2) --/
theorem closest_point_on_C₁_to_C₂ :
  ∃ (x₀ y₀ : ℝ), C₁ x₀ y₀ ∧
  (∀ (x y : ℝ), C₁ x y →
    ∀ (x' y' : ℝ), C₂ x' y' →
      distance x₀ y₀ x' y' ≤ distance x y x' y') ∧
  x₀ = -4 + Real.sqrt 2 ∧
  y₀ = 3 - Real.sqrt 2 := by
  sorry

#check closest_point_on_C₁_to_C₂

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_C₁_to_C₂_l923_92385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_yellow_balls_l923_92358

theorem extra_yellow_balls (total : ℕ) (ratio_num : ℕ) (ratio_denom : ℕ) : 
  total = 64 →
  ratio_num = 8 →
  ratio_denom = 13 →
  ∃ (white yellow extra : ℕ),
    white = yellow ∧
    white + yellow = total ∧
    (white : ℚ) / ((yellow : ℚ) + (extra : ℚ)) = (ratio_num : ℚ) / (ratio_denom : ℚ) ∧
    extra = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_yellow_balls_l923_92358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l923_92387

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / x
def g (x : ℝ) : ℝ := -x - Real.log (-x)

-- State the theorem
theorem function_inequality (a : ℝ) :
  a ≠ 0 →
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (h : ℝ), 0 < |h| ∧ |h| < ε → f a (1 + h) ≤ f a 1) →
  (∀ (x₁ : ℝ), x₁ ∈ Set.Icc 1 2 → ∃ (x₂ : ℝ), x₂ ∈ Set.Icc (-3) (-2) ∧ f a x₁ ≥ g x₂) →
  -2 < a →
  a < 0 →
  a ∈ Set.Ioo (-2) (-1 - 1/2 * Real.log 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l923_92387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sachin_age_is_14_l923_92388

-- Define Sachin's and Rahul's ages
def sachin_age : ℕ := sorry
def rahul_age : ℕ := sorry

-- Define the conditions
axiom age_difference : rahul_age = sachin_age + 4
axiom age_ratio : (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9

-- Theorem to prove
theorem sachin_age_is_14 : sachin_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sachin_age_is_14_l923_92388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_3x_plus_2_l923_92339

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 2)
def domain_f_x2_minus_2 : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem domain_f_3x_plus_2 (h : ∀ x ∈ domain_f_x2_minus_2, f (x^2 - 2) = f (x^2 - 2)) :
  {x : ℝ | f (3*x + 2) = f (3*x + 2)} = Set.Icc (-4/3) (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_3x_plus_2_l923_92339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_E_l923_92319

noncomputable def E (a b c : ℝ) : ℝ := a^3 / (1 - a^2) + b^3 / (1 - b^2) + c^3 / (1 - c^2)

theorem smallest_value_of_E (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  E a b c ≥ 1/8 ∧ (E a b c = 1/8 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_E_l923_92319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_perimeter_l923_92361

/-- Definition of a hyperbola -/
def is_hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of a point being on a hyperbola -/
def on_hyperbola (p : ℝ × ℝ) (a b : ℝ) : Prop :=
  is_hyperbola p.1 p.2 a b

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem about the perimeter of a triangle formed by a chord through a focus of a hyperbola -/
theorem hyperbola_chord_perimeter
  (a b : ℝ) (A B F₁ F₂ : ℝ × ℝ) (m : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hA : on_hyperbola A a b)
  (hB : on_hyperbola B a b)
  (hF₁ : F₁ = (Real.sqrt (a^2 + b^2), 0) ∨ F₁ = (-Real.sqrt (a^2 + b^2), 0))
  (hF₂ : F₂ = (Real.sqrt (a^2 + b^2), 0) ∨ F₂ = (-Real.sqrt (a^2 + b^2), 0))
  (hF₁F₂ : F₁ ≠ F₂)
  (hm : distance A B = m)
  (hAF₁ : distance A F₁ + distance B F₁ = m) :
  distance A F₂ + distance B F₂ + m = 4 * a + 2 * m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_perimeter_l923_92361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l923_92330

-- Define the lower bound function
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt x

-- Define the upper bound function
def g (x : ℝ) : ℝ := x^2

-- State the theorem
theorem area_between_curves : 
  ∫ x in (0)..(1), g x - f x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l923_92330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_iff_opposite_angles_supplementary_l923_92313

-- Define a point in 2D space
structure Point :=
  (x y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of being inscribed in a circle
def is_inscribed (q : Quadrilateral) : Prop := sorry

-- Define the angle measure function
noncomputable def angle_measure (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_inscribed_iff_opposite_angles_supplementary (q : Quadrilateral) :
  is_inscribed q ↔ 
    angle_measure q.A q.B q.C + angle_measure q.C q.D q.A = 180 ∧
    angle_measure q.B q.C q.D + angle_measure q.D q.A q.B = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_iff_opposite_angles_supplementary_l923_92313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l923_92324

theorem evaluate_expression (a : ℝ) : (a + 9) - a + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l923_92324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_geometric_l923_92362

/-- ArithmeticSeq x a b y means x, a, b, y form an arithmetic sequence -/
def ArithmeticSeq (x a b y : ℝ) : Prop :=
  ∃ r : ℝ, a = x + r ∧ b = a + r ∧ y = b + r

/-- GeometricSeq x c d y means x, c, d, y form a geometric sequence -/
def GeometricSeq (x c d y : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ c = x * r ∧ d = c * r ∧ y = d * r

/-- Given x > 0, y > 0, and that x, a, b, y form an arithmetic sequence,
    while x, c, d, y form a geometric sequence,
    the minimum value of (a+b)²/(cd) is 4 -/
theorem min_value_arithmetic_geometric (x y a b c d : ℝ) 
    (hx : x > 0) (hy : y > 0)
    (h_arith : ArithmeticSeq x a b y)
    (h_geom : GeometricSeq x c d y) :
    (a + b)^2 / (c * d) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_arithmetic_geometric_l923_92362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_after_seven_nonprimes_l923_92352

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_sequence_of_seven_nonprimes (n : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → ¬(is_prime (n + i))

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧
  (is_sequence_of_seven_nonprimes 90) ∧
  (∀ m : ℕ, m < 97 → ¬(is_prime m ∧ ∃ n : ℕ, n < m ∧ is_sequence_of_seven_nonprimes n)) :=
by sorry

#check smallest_prime_after_seven_nonprimes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_after_seven_nonprimes_l923_92352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_30_l923_92396

-- Define the triangle vertices
def A : ℚ × ℚ := (0, 0)
def B : ℚ × ℚ := (8, -3)
def C : ℚ × ℚ := (4, 7)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem triangle_area_is_30 :
  triangleArea A B C = 30 := by
  -- Unfold definitions and simplify
  unfold triangleArea A B C
  simp [abs_of_nonneg]
  -- The rest of the proof would go here
  sorry

#eval triangleArea A B C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_30_l923_92396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l923_92323

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2 - 1
def curve2 (x : ℝ) : ℝ := 2 - 2*x^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(1, 0), (-1, 0)}

-- Define the symmetry property
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the area calculation function
noncomputable def area_between_curves (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, g x - f x

-- Theorem statement
theorem area_enclosed_by_curves :
  symmetric_about_y_axis curve1 ∧
  symmetric_about_y_axis curve2 ∧
  intersection_points = {(1, 0), (-1, 0)} →
  area_between_curves curve1 curve2 (-1) 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l923_92323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_one_minus_two_i_l923_92376

/-- The complex number i such that i^2 = -1 -/
def i : ℂ := Complex.I

/-- The complex fraction (3 - i) / (1 + i) -/
noncomputable def complex_fraction : ℂ := (3 - i) / (1 + i)

/-- Theorem stating that the complex fraction (3 - i) / (1 + i) equals 1 - 2i -/
theorem complex_fraction_equals_one_minus_two_i : 
  complex_fraction = 1 - 2 * i := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equals_one_minus_two_i_l923_92376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_students_count_l923_92332

/-- Represents a snore-meter reading -/
def SnoreMeterReading := ℕ

/-- Represents the classroom setup -/
structure Classroom where
  side_length : ℝ
  is_regular_hexagon : Bool
  snore_meters : Fin 6 → SnoreMeterReading
  detection_radius : ℝ

/-- Calculates the total reading from all snore-meters -/
def total_reading (c : Classroom) : ℕ :=
  (Finset.range 6).sum (fun i => c.snore_meters i)

/-- Determines if a classroom setup is valid according to the problem conditions -/
def is_valid_classroom (c : Classroom) : Prop :=
  c.side_length = 3 ∧ c.is_regular_hexagon ∧ c.detection_radius = 3

/-- Represents the number of sleeping students in the classroom -/
def number_of_sleeping_students (c : Classroom) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem sleeping_students_count (c : Classroom) 
  (h_valid : is_valid_classroom c) 
  (h_total : total_reading c = 7) : 
  ∃ (n : ℕ), n = 3 ∧ n = number_of_sleeping_students c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_students_count_l923_92332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l923_92318

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

noncomputable def b (a₁ d : ℝ) : ℕ → ℝ
| 0 => 49  -- Added case for 0
| 1 => 49
| (n + 2) => 18 * (n + 2 : ℝ) - 69

theorem arithmetic_sequence_solution (a₁ d : ℝ) :
  (S a₁ d 5) * (S a₁ d 6) + 15 = 0 ∧ S a₁ d 5 ≠ 5 →
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 10 - 3 * n) ∧
  (∀ n : ℕ, b a₁ d n = (arithmetic_sequence a₁ d n)^2 - 
    if n = 1 then 0 else (arithmetic_sequence a₁ d (n-1))^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l923_92318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_f_odd_range_l923_92382

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: f is an increasing function
theorem f_increasing (a : ℝ) : 
  ∀ x y, x < y → f a x < f a y := by
  sorry

-- Theorem 2: f is an odd function if and only if a = 1
theorem f_odd_iff_a_eq_one (a : ℝ) : 
  (∀ x, f a (-x) = -(f a x)) ↔ a = 1 := by
  sorry

-- Theorem 3: When f is an odd function, its range is (-1, 1)
theorem f_odd_range : 
  Set.range (f 1) = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_f_odd_range_l923_92382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_100_l923_92321

open BigOperators

def seriesSum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (3 + 4 * (k + 1 : ℚ)) / (3^(n - k))

theorem series_sum_100 :
  seriesSum 100 = 405/2 - 5/(2 * 3^99) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_100_l923_92321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l923_92312

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + bridge_length
  total_distance / train_speed_mps

/-- Theorem: The time taken for a train of length 120 meters, moving at 54 kmph,
    to cross a bridge of length 660 meters is 52 seconds -/
theorem train_crossing_bridge_time :
  train_crossing_time 120 54 660 = 52 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval train_crossing_time 120 54 660

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l923_92312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l923_92370

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = 3, c = 1, and A = 2B, prove that cos B = √3/3 -/
theorem triangle_cosine_value (a b c A B C : ℝ) : 
  b = 3 → c = 1 → A = 2 * B → Real.cos B = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_value_l923_92370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_roots_of_unity_l923_92363

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- The set of roots of f -/
def roots_of_f : Set ℂ := {z : ℂ | f z = 0}

/-- n-th roots of unity -/
def nth_roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

/-- The statement that all roots of f are n-th roots of unity -/
def all_roots_are_nth_roots (n : ℕ) : Prop :=
  roots_of_f ⊆ nth_roots_of_unity n

/-- The proposition that n is the smallest positive integer satisfying the condition -/
def is_smallest_n (n : ℕ) : Prop :=
  all_roots_are_nth_roots n ∧ n > 0 ∧ ∀ m, m < n → ¬(all_roots_are_nth_roots m ∧ m > 0)

theorem smallest_n_for_roots_of_unity :
  is_smallest_n 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_roots_of_unity_l923_92363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_proof_l923_92399

theorem trigonometric_ratio_proof (θ : Real) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.cos θ = Real.sqrt 10 / 10) : 
  (Real.cos (2*θ)) / (Real.sin (2*θ) + (Real.cos θ)^2) = -8/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_proof_l923_92399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l923_92393

theorem vector_difference_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (-2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 10 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l923_92393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_perpendicular_points_l923_92341

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the foci of the ellipse
def foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  F₁ = (2, 0) ∧ F₂ = (-2, 0)

-- Define the perpendicular condition
def perpendicular_condition (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Theorem statement
theorem two_perpendicular_points :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ P ∈ S, ellipse P.1 P.2) ∧ 
    (∀ P ∈ S, ∃ F₁ F₂, foci F₁ F₂ ∧ perpendicular_condition P F₁ F₂) ∧
    (∃ (f : S → Fin 2), Function.Bijective f) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_perpendicular_points_l923_92341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_propositions_l923_92322

theorem two_correct_propositions :
  (∀ x : ℝ, x^2 + 1/4 ≥ x) ∧
  (∃ x : ℝ, x ∉ {k * Real.pi | k : ℤ} ∧ Real.sin x + 1 / Real.sin x < 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y) * (1/x + 4/y) < 8) ∧
  (∀ x : ℝ, x > 1 → x + 1/(x-1) ≥ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_propositions_l923_92322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l923_92302

/-- The focus of a parabola y^2 = 4x is at the point (1,0) -/
theorem parabola_focus (x y : ℝ) : 
  y^2 = 4*x → (1, 0) = (x + 1, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l923_92302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l923_92325

theorem tan_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 1/3) (h2 : π/2 < α ∧ α < π) :
  Real.tan (π - α) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l923_92325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_is_88_l923_92346

noncomputable def sphere_radius : ℝ := 3 * (36 / Real.pi)

def prism_length : ℝ := 6
def prism_width : ℝ := 4

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def prism_volume (l w h : ℝ) : ℝ := l * w * h

def prism_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

theorem prism_surface_area_is_88 :
  ∃ (h : ℝ), 
    sphere_volume sphere_radius = prism_volume prism_length prism_width h →
    prism_surface_area prism_length prism_width h = 88 :=
by
  -- Introduce the height variable
  let h := 2

  -- Show that this height satisfies the volume equality
  have volume_eq : sphere_volume sphere_radius = prism_volume prism_length prism_width h := by
    sorry  -- The actual proof would go here

  -- Calculate the surface area
  have surface_area : prism_surface_area prism_length prism_width h = 88 := by
    sorry  -- The actual calculation would go here

  -- Conclude the proof
  exact ⟨h, fun _ => surface_area⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_is_88_l923_92346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_weights_l923_92373

def chihuahua_weight : ℝ → Prop := sorry
def pitbull_weight : ℝ → Prop := sorry
def great_dane_weight : ℝ → Prop := sorry

theorem dog_weights (c p g : ℝ) 
  (h1 : chihuahua_weight c)
  (h2 : pitbull_weight p)
  (h3 : great_dane_weight g)
  (combined_weight : c + p + g = 439)
  (pitbull_ratio : p = 3 * c)
  (great_dane_ratio : g = 3 * p + 10) :
  g = 307 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_weights_l923_92373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_10pi_minus_theta_l923_92381

theorem tan_10pi_minus_theta (θ : ℝ) 
  (h1 : π < θ) (h2 : θ < 2*π) (h3 : Real.cos (θ - 9*π) = -3/5) : 
  Real.tan (10*π - θ) = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_10pi_minus_theta_l923_92381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_gon_coloring_invariance_l923_92377

/-- Represents a segment in the n-gon -/
structure Segment where
  start : ℕ
  end_ : ℕ
  value : ℝ

/-- Represents the state of the n-gon -/
structure NPolygonState where
  n : ℕ
  segments : List Segment

/-- Represents a move on the n-gon -/
def move (state : NPolygonState) (a b c d : ℕ) : NPolygonState :=
  sorry

/-- Checks if two states are equivalent (same colored segments) -/
def equivalent_states (s1 s2 : NPolygonState) : Prop :=
  sorry

theorem n_gon_coloring_invariance (n : ℕ) (initial_state : NPolygonState) 
    (final_state : NPolygonState) (h1 : initial_state.n = n) 
    (h2 : final_state.n = n) (h3 : initial_state.segments.length = 2 * n - 3) 
    (h4 : ∀ s ∈ initial_state.segments, s.value > 0) 
    (h5 : equivalent_states initial_state final_state) : 
  ∀ s ∈ initial_state.segments, ∃ s' ∈ final_state.segments, s = s' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_gon_coloring_invariance_l923_92377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l923_92378

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Represents the parallelogram ABCD -/
def parallelogram : (Point × Point × Point × Point) :=
  (Point.mk 0 0, Point.mk 2 3, Point.mk 5 3, Point.mk 3 0)

/-- The midpoint of diagonal AC -/
noncomputable def E : Point :=
  let (A, _, C, _) := parallelogram
  Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2)

/-- Point F on side DA such that DF = 2/3 DA -/
noncomputable def F : Point :=
  let (A, _, _, D) := parallelogram
  Point.mk (A.x + 2/3 * (D.x - A.x)) (A.y + 2/3 * (D.y - A.y))

theorem area_ratio_theorem : 
  let (A, B, _, D) := parallelogram
  let areaDFE := triangleArea D F E
  let areaABE := triangleArea A B E
  let areaAEF := triangleArea A E F
  let areaABEF := areaABE + areaAEF
  areaDFE / areaABEF = 50 / 32.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l923_92378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l923_92374

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define the area of a triangle
noncomputable def area_triangle (A K F : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((A.1 - K.1)^2 + (A.2 - K.2)^2)
  let b := Real.sqrt ((K.1 - F.1)^2 + (K.2 - F.2)^2)
  let c := Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the theorem
theorem parabola_triangle_area 
  (A B K : ℝ × ℝ) 
  (hA : parabola A.1 A.2)
  (hB : directrix B.1)
  (hK : directrix K.1)
  (hFK : K.2 = 0) -- K is on x-axis
  (hAK_perp : (A.2 - K.2) * (B.1 - K.1) = 0) -- AK ⟂ directrix
  (hAF_eq_BF : (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = 
               (B.1 - focus.1)^2 + (B.2 - focus.2)^2) :
  area_triangle A K focus = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l923_92374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_files_after_deletion_l923_92375

/-- Represents the number of files on a flash drive -/
structure FlashDrive where
  music : Nat
  video : Nat
  document : Nat

/-- Calculates the total number of files left after deletion -/
def totalFilesAfterDeletion (drive1 drive2 : FlashDrive) (deletedMusic deletedVideo : Nat) : Nat :=
  (drive1.music + drive2.music - deletedMusic) +
  (drive1.video + drive2.video - deletedVideo) +
  (drive1.document + drive2.document)

theorem files_after_deletion :
  let drive1 : FlashDrive := { music := 150, video := 235, document := 75 }
  let drive2 : FlashDrive := { music := 90, video := 285, document := 40 }
  let deletedMusic := 45
  let deletedVideo := 200
  totalFilesAfterDeletion drive1 drive2 deletedMusic deletedVideo = 630 := by
  sorry

#eval totalFilesAfterDeletion
  { music := 150, video := 235, document := 75 }
  { music := 90, video := 285, document := 40 }
  45
  200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_files_after_deletion_l923_92375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisors_2_to_20_l923_92311

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

def divisorCount (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def evenNumbersFrom2To20 : Finset ℕ := 
  Finset.filter (fun n => n % 2 = 0 ∧ 2 ≤ n ∧ n ≤ 20) (Finset.range 21)

theorem greatest_divisors_2_to_20 :
  ∀ n ∈ evenNumbersFrom2To20, divisorCount n ≤ 6 ∧
  (n = 12 ∨ n = 18 ∨ n = 20 ↔ divisorCount n = 6) := by
  sorry

#eval evenNumbersFrom2To20
#eval divisorCount 12
#eval divisorCount 18
#eval divisorCount 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisors_2_to_20_l923_92311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hkmo_problem_l923_92351

theorem hkmo_problem (total_teams : ℕ) 
  (solved_q1 solved_q2 solved_q3 solved_q4 : ℕ) : 
  total_teams = 50 →
  solved_q1 = 45 →
  solved_q2 = 40 →
  solved_q3 = 35 →
  solved_q4 = 30 →
  (∀ t : ℕ, t ≤ total_teams → t > 0 → ∃ q : ℕ, q ≤ 4 ∧ q > 0 ∧ t ∉ Finset.range q) →
  (solved_q1 + solved_q2 + solved_q3 + solved_q4 = 3 * total_teams) →
  ∃ x : ℕ, (total_teams - solved_q1) + (total_teams - solved_q2) = 
    (total_teams - (solved_q3 + solved_q4 - x)) ∧ x = 15 :=
by
  sorry

#check hkmo_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hkmo_problem_l923_92351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l923_92342

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = Real.sqrt 6 / 3 →
  Real.cos B = 2 * Real.sqrt 2 / 3 →
  c = 2 * Real.sqrt 2 →
  (1 / 2) * a * c * Real.sin B = 2 * Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l923_92342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_shifted_not_in_second_quadrant_l923_92379

/-- A function f : ℝ → ℝ is said to pass through a quadrant if there exists a point (x, y) in that quadrant such that y = f(x) -/
def passes_through_quadrant (f : ℝ → ℝ) (quad : Nat) : Prop :=
  match quad with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ f x = y
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ f x = y
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ f x = y
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ f x = y
  | _ => False

/-- The exponential function with base a and vertical shift b -/
noncomputable def exp_shifted (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem exp_shifted_not_in_second_quadrant (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ¬(passes_through_quadrant (exp_shifted a b) 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_shifted_not_in_second_quadrant_l923_92379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l923_92368

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  (floor x : ℝ) - 2 * x

-- Theorem statement
theorem range_of_g :
  Set.range g = Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l923_92368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antonella_coins_l923_92307

/-- Represents the types of coins Antonella has --/
inductive Coin
  | Loonie
  | Toonie

/-- The value of a coin in dollars --/
def coinValue : Coin → Nat
  | Coin.Loonie => 1
  | Coin.Toonie => 2

/-- Antonella's initial coin collection --/
structure CoinCollection where
  loonies : Nat
  toonies : Nat

/-- Calculate the total value of coins in the collection --/
def CoinCollection.totalValue (c : CoinCollection) : Nat :=
  c.loonies * coinValue Coin.Loonie + c.toonies * coinValue Coin.Toonie

/-- Calculate the total number of coins in the collection --/
def CoinCollection.totalCoins (c : CoinCollection) : Nat :=
  c.loonies + c.toonies

/-- Theorem: Antonella has 10 coins in total --/
theorem antonella_coins (c : CoinCollection) 
  (h1 : c.toonies = 4)
  (h2 : c.totalValue = 14)
  (h3 : c.totalValue - 3 = 11) : 
  c.totalCoins = 10 := by
  sorry

#check antonella_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antonella_coins_l923_92307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l923_92300

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - 4*x + a^2) / Real.log 10

def p (a : ℝ) : Prop := ∀ x, ∃ y, f a x = y

def q (a : ℝ) : Prop := ∀ m ∈ Set.Icc (-1) 1, a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (a ∈ Set.Icc (-2) (-1) ∨ a ∈ Set.Ioo 2 6) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l923_92300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_is_correct_l923_92364

/-- The side length of the largest square that can be inscribed in a 15x15 square 
    with two congruent equilateral triangles drawn inside, sharing one side and 
    each having one vertex on a vertex of the square. -/
noncomputable def largest_inscribed_square_side_length : ℝ := (15 - 5 * Real.sqrt 3) / 3

/-- Theorem stating that the calculated side length is correct -/
theorem largest_inscribed_square_side_length_is_correct 
  (outer_square_side : ℝ) 
  (triangle_side : ℝ) 
  (h_outer_square : outer_square_side = 15) 
  (h_triangle : triangle_side = 15 * Real.sqrt 6 / 3) : 
  largest_inscribed_square_side_length = (15 - 5 * Real.sqrt 3) / 3 := by
  sorry

#check largest_inscribed_square_side_length_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_is_correct_l923_92364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_integer_in_sequence_l923_92348

theorem tenth_integer_in_sequence (seq : List ℤ) : 
  seq.length = 20 ∧ 
  (∀ i : ℕ, i < 19 → seq[i+1]! = seq[i]! + 1) ∧
  (seq.sum : ℚ) / 20 = 23.65 →
  seq[9]! = 23 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_integer_in_sequence_l923_92348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l923_92336

noncomputable section

-- Define the circles
def circle1_center : ℝ × ℝ := (0, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (12, 0)
def circle2_radius : ℝ := 5

-- Define the tangent point
def tangent_point : ℝ := 9/2

-- Theorem statement
theorem tangent_line_intersection :
  ∃ (y : ℝ), 
    -- The point (tangent_point, 0) is to the right of the origin
    tangent_point > 0 ∧
    -- The distance from (tangent_point, 0) to circle1_center is equal to the radius of circle1
    Real.sqrt ((tangent_point - circle1_center.1) ^ 2 + (0 - circle1_center.2) ^ 2) = circle1_radius ∧
    -- The distance from (tangent_point, 0) to circle2_center is equal to the radius of circle2
    Real.sqrt ((tangent_point - circle2_center.1) ^ 2 + (0 - circle2_center.2) ^ 2) = circle2_radius ∧
    -- The line through (tangent_point, 0) and (tangent_point, y) is tangent to both circles
    ((tangent_point - circle1_center.1) * (y - circle1_center.2) = 
     circle1_radius * Real.sqrt ((tangent_point - circle1_center.1) ^ 2 + (y - circle1_center.2) ^ 2)) ∧
    ((tangent_point - circle2_center.1) * (y - circle2_center.2) = 
     circle2_radius * Real.sqrt ((tangent_point - circle2_center.1) ^ 2 + (y - circle2_center.2) ^ 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l923_92336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_result_l923_92365

/-- Applies dilation to a complex number -/
def dilate (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

/-- Applies 90° counterclockwise rotation to a complex number -/
def rotate90 (center : ℂ) (z : ℂ) : ℂ :=
  center + Complex.I * (z - center)

/-- Combines dilation and rotation -/
def dilateAndRotate (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  rotate90 center (dilate center scale z)

theorem dilation_rotation_result :
  let center : ℂ := 2 + 3 * Complex.I
  let initial : ℂ := -1 + Complex.I
  let scale : ℝ := 3
  let result : ℂ := dilateAndRotate center scale initial
  result = -4 + 12 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_result_l923_92365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_lot_count_l923_92317

theorem car_lot_count 
  (power_steering : ℕ)
  (power_windows : ℕ)
  (both_features : ℕ)
  (neither_feature : ℕ)
  (h1 : power_steering = 45)
  (h2 : power_windows = 25)
  (h3 : both_features = 17)
  (h4 : neither_feature = 12) :
  power_steering + power_windows - both_features + neither_feature = 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_lot_count_l923_92317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_tails_with_second_head_l923_92340

/-- A fair coin flip can result in either heads or tails with equal probability. -/
def FairCoin : Type := Bool

/-- The outcome of flipping a fair coin multiple times until getting two consecutive
    heads or two consecutive tails. -/
inductive FlipOutcome
  | TwoHeads
  | TwoTails

/-- The probability of getting two tails in a row but seeing a second head before 
    seeing a second tail when flipping a fair coin repeatedly until getting either 
    two heads or two tails in a row. -/
noncomputable def probTwoTailsWithSecondHead : ℝ := 1 / 24

/-- Theorem stating that the probability of getting two tails in a row but seeing 
    a second head before seeing a second tail when flipping a fair coin repeatedly 
    until getting either two heads or two tails in a row is 1/24. -/
theorem prob_two_tails_with_second_head :
  probTwoTailsWithSecondHead = 1 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_tails_with_second_head_l923_92340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l923_92359

def orthogonal_unit_vectors (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ (a.1^2 + a.2^2 = 1) ∧ (b.1^2 + b.2^2 = 1)

theorem vector_magnitude_range (a b c : ℝ × ℝ) 
  (h1 : orthogonal_unit_vectors a b) 
  (h2 : (c.1 - a.1 - b.1)^2 + (c.2 - a.2 - b.2)^2 = 1) : 
  Real.sqrt 2 - 1 ≤ Real.sqrt (c.1^2 + c.2^2) ∧ Real.sqrt (c.1^2 + c.2^2) ≤ Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l923_92359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_domain_l923_92395

noncomputable section

/-- The function f(x) = √(4 - 3x - x^2) -/
def f (x : ℝ) : ℝ := Real.sqrt (4 - 3*x - x^2)

/-- The domain D of f(x) -/
def D : Set ℝ := {x | 4 - 3*x - x^2 ≥ 0}

/-- The interval [-5, 5] -/
def I : Set ℝ := Set.Icc (-5) 5

theorem probability_in_domain :
  (MeasureTheory.volume (D ∩ I)) / (MeasureTheory.volume I) = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_domain_l923_92395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_value_l923_92355

theorem least_integer_value (a b c d e f : ℤ) : 
  a < b → b < c → c < d → d < e → e < f →  -- 6 different integers
  (a + b + c + d + e + f) / 6 = 85 →        -- average is 85
  f = 90 →                                  -- largest is 90
  a ≥ 70 →                                  -- smallest is at least 70
  ∀ x : ℤ, x < 70 → 
    ¬∃ y z w v : ℤ, x < y ∧ y < z ∧ z < w ∧ w < v ∧ v < 90 
      ∧ (x + y + z + w + v + 90) / 6 = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_value_l923_92355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_star_operation_l923_92310

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {2, 4, 5}

def star_operation (A B : Finset ℕ) : Finset ℕ := A \ B

theorem subsets_of_star_operation :
  Finset.card (Finset.powerset (star_operation A B)) = 4 := by
  -- Compute A * B
  have h1 : star_operation A B = {1, 3} := by rfl
  
  -- Count the number of subsets
  have h2 : Finset.card (Finset.powerset {1, 3}) = 4 := by rfl
  
  -- Use the previous facts to prove the theorem
  rw [h1, h2]

#eval Finset.card (Finset.powerset (star_operation A B))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_star_operation_l923_92310
