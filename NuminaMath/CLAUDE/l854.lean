import Mathlib

namespace fundraising_amount_proof_l854_85461

/-- Calculates the amount each person needs to raise in a fundraising event -/
def amount_per_person (total_goal : ℕ) (initial_donation : ℕ) (num_participants : ℕ) : ℕ :=
  (total_goal - initial_donation) / num_participants

/-- Proves that given the specified conditions, each person needs to raise $225 -/
theorem fundraising_amount_proof (total_goal : ℕ) (initial_donation : ℕ) (num_participants : ℕ) 
  (h1 : total_goal = 2000)
  (h2 : initial_donation = 200)
  (h3 : num_participants = 8) :
  amount_per_person total_goal initial_donation num_participants = 225 := by
  sorry

#eval amount_per_person 2000 200 8

end fundraising_amount_proof_l854_85461


namespace select_and_swap_count_l854_85482

def num_people : ℕ := 8
def num_selected : ℕ := 3

def ways_to_select_and_swap : ℕ := Nat.choose num_people num_selected * (Nat.factorial 2)

theorem select_and_swap_count :
  ways_to_select_and_swap = Nat.choose num_people num_selected * (Nat.factorial 2) :=
by sorry

end select_and_swap_count_l854_85482


namespace inscribed_circle_radius_when_area_twice_perimeter_l854_85400

/-- A triangle with an inscribed circle -/
structure Triangle :=
  (area : ℝ)
  (perimeter : ℝ)
  (inradius : ℝ)

/-- The theorem stating that for a triangle where the area is twice the perimeter, 
    the radius of the inscribed circle is 4 -/
theorem inscribed_circle_radius_when_area_twice_perimeter (t : Triangle) 
  (h : t.area = 2 * t.perimeter) : t.inradius = 4 := by
  sorry

end inscribed_circle_radius_when_area_twice_perimeter_l854_85400


namespace race_table_distance_l854_85424

/-- Given a race with 11 equally spaced tables over 2100 meters, 
    the distance between the first and third table is 420 meters. -/
theorem race_table_distance (total_distance : ℝ) (num_tables : ℕ) :
  total_distance = 2100 →
  num_tables = 11 →
  (2 * (total_distance / (num_tables - 1))) = 420 := by
  sorry

end race_table_distance_l854_85424


namespace victor_lost_lives_l854_85410

theorem victor_lost_lives (current_lives : ℕ) (difference : ℕ) (lost_lives : ℕ) : 
  current_lives = 2 → difference = 12 → lost_lives - current_lives = difference → lost_lives = 14 := by
  sorry

end victor_lost_lives_l854_85410


namespace inscribed_iff_side_length_le_l854_85478

/-- A regular polygon -/
structure RegularPolygon where
  n : ℕ
  sideLength : ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a regular polygon is inscribed in a circle -/
def isInscribed (p : RegularPolygon) (c : Circle) : Prop :=
  sorry

/-- The side length of an inscribed regular n-gon in a given circle -/
def inscribedSideLength (n : ℕ) (c : Circle) : ℝ :=
  sorry

theorem inscribed_iff_side_length_le
  (n : ℕ) (c : Circle) (p : RegularPolygon) 
  (h1 : p.n = n) :
  isInscribed p c ↔ p.sideLength ≤ inscribedSideLength n c :=
sorry

end inscribed_iff_side_length_le_l854_85478


namespace complex_magnitude_problem_l854_85459

theorem complex_magnitude_problem (z : ℂ) (h : (z + 1) / (z - 2) = 1 - 3*I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l854_85459


namespace shopping_money_l854_85434

theorem shopping_money (initial_amount : ℝ) (remaining_amount : ℝ) : 
  remaining_amount = 140 →
  remaining_amount = initial_amount * (1 - 0.3) →
  initial_amount = 200 := by
sorry

end shopping_money_l854_85434


namespace product_of_g_at_roots_of_f_l854_85401

def f (y : ℝ) : ℝ := y^4 - y^3 + 2*y - 1

def g (y : ℝ) : ℝ := y^2 + y - 3

theorem product_of_g_at_roots_of_f :
  ∀ y₁ y₂ y₃ y₄ : ℝ,
  f y₁ = 0 → f y₂ = 0 → f y₃ = 0 → f y₄ = 0 →
  ∃ result : ℝ, g y₁ * g y₂ * g y₃ * g y₄ = result :=
by sorry

end product_of_g_at_roots_of_f_l854_85401


namespace sum_of_cubes_l854_85495

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a * b + a * c + b * c = -4) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = -5 := by
sorry

end sum_of_cubes_l854_85495


namespace bond_energy_OF_bond_energy_OF_proof_l854_85447

-- Define the molecules and atoms
inductive Molecule | OF₂ | O₂ | F₂
inductive Atom | O | F

-- Define the enthalpy of formation for OF₂
def enthalpy_formation_OF₂ : ℝ := 22

-- Define the bond energies for O₂ and F₂
def bond_energy_O₂ : ℝ := 498
def bond_energy_F₂ : ℝ := 159

-- Define the thermochemical equations
def thermochem_OF₂ (x : ℝ) : Prop :=
  x = 1 * bond_energy_F₂ + 0.5 * bond_energy_O₂ - enthalpy_formation_OF₂

-- Theorem: The bond energy of O-F in OF₂ is 215 kJ/mol
theorem bond_energy_OF : ℝ :=
  215

-- Proof of the theorem
theorem bond_energy_OF_proof : 
  thermochem_OF₂ bond_energy_OF := by
  sorry


end bond_energy_OF_bond_energy_OF_proof_l854_85447


namespace thirty_percent_less_than_ninety_l854_85407

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 50.4 → x + (1/4 * x) = 90 - (30/100 * 90) := by
  sorry

end thirty_percent_less_than_ninety_l854_85407


namespace sum_of_roots_f_2y_eq_10_l854_85462

/-- The function f as defined in the problem -/
def f (x : ℝ) : ℝ := (3*x)^2 + 3*x + 1

/-- The theorem stating the sum of roots of f(2y) = 10 -/
theorem sum_of_roots_f_2y_eq_10 :
  ∃ y₁ y₂ : ℝ, f (2*y₁) = 10 ∧ f (2*y₂) = 10 ∧ y₁ + y₂ = -0.17 := by
  sorry

end sum_of_roots_f_2y_eq_10_l854_85462


namespace least_positive_angle_theorem_l854_85444

theorem least_positive_angle_theorem (θ : Real) : 
  (θ > 0 ∧ θ ≤ π / 2) → 
  (Real.cos (10 * π / 180) = Real.sin (20 * π / 180) + Real.sin θ) → 
  θ = 40 * π / 180 := by
sorry

end least_positive_angle_theorem_l854_85444


namespace unique_quadratic_solution_l854_85479

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →
  a + c = 45 →
  a < c →
  (a = (45 - 15 * Real.sqrt 5) / 2 ∧ c = (45 + 15 * Real.sqrt 5) / 2) :=
by sorry

end unique_quadratic_solution_l854_85479


namespace imaginary_part_of_complex_reciprocal_l854_85445

theorem imaginary_part_of_complex_reciprocal (z : ℂ) (h : z = -2 + I) :
  (1 / z).im = -1 / 5 := by sorry

end imaginary_part_of_complex_reciprocal_l854_85445


namespace function_properties_quadratic_inequality_solution_set_maximum_value_of_fraction_l854_85454

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem function_properties :
  ∀ a b : ℝ,
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  ∀ x, f a b x = -3 * x^2 - 3 * x + 15 :=
sorry

theorem quadratic_inequality_solution_set :
  ∀ a b c : ℝ,
  (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) →
  c ≤ -25/12 :=
sorry

theorem maximum_value_of_fraction :
  ∀ x : ℝ,
  x > -1 →
  (f (-3) 5 x - 21) / (x + 1) ≤ -3 :=
sorry

end function_properties_quadratic_inequality_solution_set_maximum_value_of_fraction_l854_85454


namespace divisors_of_square_l854_85481

theorem divisors_of_square (n : ℕ) : 
  (∃ p : ℕ, Prime p ∧ n = p^3) → 
  (Finset.card (Nat.divisors n) = 4) → 
  (Finset.card (Nat.divisors (n^2)) = 7) := by
sorry

end divisors_of_square_l854_85481


namespace product_plus_one_composite_l854_85416

theorem product_plus_one_composite : 
  ∃ (a b : ℤ), b > 1 ∧ 2014 * 2015 * 2016 * 2017 + 1 = a * b := by
  sorry

end product_plus_one_composite_l854_85416


namespace circle_line_distance_range_l854_85438

theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), (p.1 - 1)^2 + (p.2 - 1)^2 = 4 ∧ 
                      (q.1 - 1)^2 + (q.2 - 1)^2 = 4 ∧ 
                      p ≠ q ∧
                      (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 4 → 
                        (|y - (x + b)| / Real.sqrt 2 = 1 → (x, y) = p ∨ (x, y) = q))) →
  b ∈ Set.union (Set.Ioo (-3 * Real.sqrt 2) (-Real.sqrt 2)) 
                (Set.Ioo (Real.sqrt 2) (3 * Real.sqrt 2)) :=
by sorry

end circle_line_distance_range_l854_85438


namespace george_second_half_correct_l854_85458

def trivia_game (first_half_correct : ℕ) (points_per_question : ℕ) (final_score : ℕ) : ℕ :=
  (final_score - first_half_correct * points_per_question) / points_per_question

theorem george_second_half_correct :
  trivia_game 6 3 30 = 4 :=
sorry

end george_second_half_correct_l854_85458


namespace bond_value_after_eight_years_l854_85450

/-- Represents the simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.08333333333333332

theorem bond_value_after_eight_years :
  ∀ initial_investment : ℝ,
  simple_interest initial_investment interest_rate 3 = 300 →
  simple_interest initial_investment interest_rate 8 = 400 :=
by
  sorry


end bond_value_after_eight_years_l854_85450


namespace cube_sum_value_l854_85421

theorem cube_sum_value (a b R S : ℝ) : 
  a + b = R → 
  a^2 + b^2 = 12 → 
  a^3 + b^3 = S → 
  S = 32 := by
sorry

end cube_sum_value_l854_85421


namespace no_solution_to_inequalities_l854_85470

theorem no_solution_to_inequalities : 
  ¬∃ (x y : ℝ), (4*x^2 + 4*x*y + 19*y^2 ≤ 2) ∧ (x - y ≤ -1) := by
  sorry

end no_solution_to_inequalities_l854_85470


namespace ellipse_eccentricity_l854_85404

-- Define the ellipse C
def C (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def F (c : ℝ) : ℝ × ℝ := (c, 0)
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (a : ℝ) : ℝ × ℝ := (a, 0)
def P (x y : ℝ) : ℝ × ℝ := (x, y)
def M (x y : ℝ) : ℝ × ℝ := (x, y)
def E (y : ℝ) : ℝ × ℝ := (0, y)

-- Define the lines
def line_l (a c : ℝ) (x y : ℝ) : Prop := 
  ∃ (y_M : ℝ), y - y_M = (y_M / (c + a)) * (x - c)

def line_BM (a c : ℝ) (x y y_E : ℝ) : Prop := 
  y - y_E/2 = -(y_E/2) / (a + c) * x

-- Main theorem
theorem ellipse_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c < a)
  (h3 : ∃ (x y : ℝ), C a b x y ∧ P x y = (x, y) ∧ x = c)
  (h4 : ∃ (x y y_E : ℝ), line_l a c x y ∧ line_BM a c x y y_E ∧ M x y = (x, y))
  : c/a = 1/2 := by
  sorry

end ellipse_eccentricity_l854_85404


namespace perpendicular_planes_condition_l854_85492

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_condition
  (a b : Line) (α β : Plane)
  (skew : a ≠ b)
  (perp_a_α : perpendicular a α)
  (perp_b_β : perpendicular b β)
  (not_subset_a_β : ¬subset a β)
  (not_subset_b_α : ¬subset b α) :
  perpendicular_planes α β ↔ perpendicular_lines a b :=
sorry

end perpendicular_planes_condition_l854_85492


namespace sum_of_first_10_terms_equals_560_l854_85499

def arithmetic_sequence_1 (n : ℕ) : ℕ := 4 * n - 2
def arithmetic_sequence_2 (n : ℕ) : ℕ := 6 * n - 4

def common_sequence (n : ℕ) : ℕ := 12 * n - 10

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (12 * n - 8) / 2

theorem sum_of_first_10_terms_equals_560 :
  sum_of_first_n_terms 10 = 560 := by sorry

end sum_of_first_10_terms_equals_560_l854_85499


namespace shared_circles_existence_l854_85427

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to check if a point is on a circle
def isPointOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a function to check if a circle is the circumcircle of a triangle
def isCircumcircle (c : Circle) (t : Triangle) : Prop :=
  isPointOnCircle t.A c ∧ isPointOnCircle t.B c ∧ isPointOnCircle t.C c

-- Define a function to check if a circle is the inscribed circle of a triangle
def isInscribedCircle (c : Circle) (t : Triangle) : Prop :=
  -- This is a simplified condition; in reality, it would involve more complex geometric relationships
  true

-- The main theorem
theorem shared_circles_existence 
  (ABC : Triangle) 
  (O : Circle) 
  (I : Circle) 
  (h1 : isCircumcircle O ABC) 
  (h2 : isInscribedCircle I ABC) 
  (D : ℝ × ℝ) 
  (h3 : isPointOnCircle D O) : 
  ∃ (DEF : Triangle), isCircumcircle O DEF ∧ isInscribedCircle I DEF :=
sorry

end shared_circles_existence_l854_85427


namespace min_sum_squares_l854_85442

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 14}

theorem min_sum_squares (p q r s t u v w : Int) 
  (hp : p ∈ S) (hq : q ∈ S) (hr : r ∈ S) (hs : s ∈ S)
  (ht : t ∈ S) (hu : u ∈ S) (hv : v ∈ S) (hw : w ∈ S)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
               q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
               r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
               s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
               t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
               u ≠ v ∧ u ≠ w ∧
               v ≠ w) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 ∧ 
  ∃ (p' q' r' s' t' u' v' w' : Int),
    p' ∈ S ∧ q' ∈ S ∧ r' ∈ S ∧ s' ∈ S ∧ t' ∈ S ∧ u' ∈ S ∧ v' ∈ S ∧ w' ∈ S ∧
    p' ≠ q' ∧ p' ≠ r' ∧ p' ≠ s' ∧ p' ≠ t' ∧ p' ≠ u' ∧ p' ≠ v' ∧ p' ≠ w' ∧
    q' ≠ r' ∧ q' ≠ s' ∧ q' ≠ t' ∧ q' ≠ u' ∧ q' ≠ v' ∧ q' ≠ w' ∧
    r' ≠ s' ∧ r' ≠ t' ∧ r' ≠ u' ∧ r' ≠ v' ∧ r' ≠ w' ∧
    s' ≠ t' ∧ s' ≠ u' ∧ s' ≠ v' ∧ s' ≠ w' ∧
    t' ≠ u' ∧ t' ≠ v' ∧ t' ≠ w' ∧
    u' ≠ v' ∧ u' ≠ w' ∧
    v' ≠ w' ∧
    (p' + q' + r' + s')^2 + (t' + u' + v' + w')^2 = 98 :=
by sorry

end min_sum_squares_l854_85442


namespace february_highest_percentage_l854_85414

-- Define the months
inductive Month
| January
| February
| March
| April
| May

-- Define the sales data for each month
def sales_data (m : Month) : (Nat × Nat × Nat) :=
  match m with
  | Month.January => (5, 4, 6)
  | Month.February => (6, 5, 7)
  | Month.March => (5, 5, 8)
  | Month.April => (4, 6, 7)
  | Month.May => (3, 4, 5)

-- Calculate the percentage difference
def percentage_difference (m : Month) : Rat :=
  let (d, b, f) := sales_data m
  let c := d + b
  (c - f : Rat) / f * 100

-- Theorem statement
theorem february_highest_percentage :
  ∀ m : Month, m ≠ Month.February →
  percentage_difference Month.February ≥ percentage_difference m :=
by sorry

end february_highest_percentage_l854_85414


namespace f_inequality_l854_85426

/-- The function f(x) = x^2 - 2x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

/-- Theorem stating that f(0) < f(4) < f(-4) for any real c -/
theorem f_inequality (c : ℝ) : f c 0 < f c 4 ∧ f c 4 < f c (-4) := by
  sorry

end f_inequality_l854_85426


namespace parabola_focus_distance_l854_85418

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

theorem parabola_focus_distance (C : Parabola) (A : PointOnParabola C)
  (h_focus_dist : Real.sqrt ((A.x - C.p / 2)^2 + A.y^2) = 12)
  (h_y_axis_dist : A.x = 9) :
  C.p = 6 := by
  sorry

end parabola_focus_distance_l854_85418


namespace worker_a_time_l854_85425

theorem worker_a_time (worker_b_time worker_ab_time worker_a_time : ℝ) : 
  worker_b_time = 12 →
  worker_ab_time = 5.454545454545454 →
  worker_a_time = 10.153846153846153 →
  (1 / worker_a_time + 1 / worker_b_time) * worker_ab_time = 1 :=
by sorry

end worker_a_time_l854_85425


namespace weight_loss_challenge_l854_85403

theorem weight_loss_challenge (original_weight : ℝ) (x : ℝ) : 
  x > 0 →
  x < 100 →
  let final_weight := original_weight * (1 - x / 100 + 2 / 100)
  let measured_loss_percentage := 13.3
  final_weight = original_weight * (1 - measured_loss_percentage / 100) →
  x = 15.3 := by
sorry

end weight_loss_challenge_l854_85403


namespace king_of_red_suit_probability_l854_85487

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)

/-- Represents the number of Kings of red suits in a standard deck -/
structure RedKings :=
  (count : Nat)

/-- The probability of selecting a specific card from a deck -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

theorem king_of_red_suit_probability (d : Deck) (rk : RedKings) :
  d.cards = 52 → rk.count = 2 → probability rk.count d.cards = 1 / 26 := by
  sorry

end king_of_red_suit_probability_l854_85487


namespace minimize_f_l854_85440

/-- The function f(x) = x^2 + 8x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 3

/-- Theorem: The value of x that minimizes f(x) = x^2 + 8x + 3 is -4 -/
theorem minimize_f :
  ∃ (x_min : ℝ), x_min = -4 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by sorry

end minimize_f_l854_85440


namespace rectangular_prism_diagonal_l854_85496

/-- The diagonal of a rectangular prism with dimensions 15, 25, and 15 is 5√43 -/
theorem rectangular_prism_diagonal : 
  ∀ (a b c d : ℝ), 
    a = 15 → 
    b = 25 → 
    c = 15 → 
    d ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 → 
    d = 5 * Real.sqrt 43 := by
  sorry

end rectangular_prism_diagonal_l854_85496


namespace tomatoes_left_l854_85493

theorem tomatoes_left (initial_tomatoes : ℕ) (eaten_fraction : ℚ) : initial_tomatoes = 21 ∧ eaten_fraction = 1/3 → initial_tomatoes - (initial_tomatoes * eaten_fraction).floor = 14 := by
  sorry

end tomatoes_left_l854_85493


namespace smallest_x_equals_f_2003_l854_85446

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The problem statement -/
theorem smallest_x_equals_f_2003 :
  (∀ x > 0, f (3 * x) = 3 * f x) →
  (∀ x ∈ Set.Icc 2 4, f x = 1 - |x - 2|) →
  (∃ x₀ > 0, f x₀ = f 2003 ∧ ∀ x > 0, f x = f 2003 → x ≥ x₀) →
  (∃ x₀ > 0, f x₀ = f 2003 ∧ ∀ x > 0, f x = f 2003 → x ≥ x₀ ∧ x₀ = 1422817) :=
by sorry

end smallest_x_equals_f_2003_l854_85446


namespace equal_piles_coin_count_l854_85474

theorem equal_piles_coin_count (total_coins : ℕ) (num_quarter_piles : ℕ) (num_dime_piles : ℕ) :
  total_coins = 42 →
  num_quarter_piles = 3 →
  num_dime_piles = 3 →
  ∃ (coins_per_pile : ℕ),
    total_coins = coins_per_pile * (num_quarter_piles + num_dime_piles) ∧
    coins_per_pile = 7 :=
by sorry

end equal_piles_coin_count_l854_85474


namespace image_of_negative_four_two_l854_85436

/-- The mapping f from R² to R² defined by f(x, y) = (xy, x + y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 * p.2, p.1 + p.2)

/-- Theorem stating that f(-4, 2) = (-8, -2) -/
theorem image_of_negative_four_two :
  f (-4, 2) = (-8, -2) := by
  sorry

end image_of_negative_four_two_l854_85436


namespace total_fish_caught_l854_85466

/-- The number of times Chris goes fishing -/
def chris_trips : ℕ := 10

/-- The number of fish Brian catches per trip -/
def brian_fish_per_trip : ℕ := 400

/-- The ratio of Brian's fishing frequency to Chris's -/
def brian_frequency_ratio : ℚ := 2

/-- The fraction of fish Brian catches compared to Chris per trip -/
def brian_catch_fraction : ℚ := 3/5

theorem total_fish_caught :
  let brian_trips := chris_trips * brian_frequency_ratio
  let chris_fish_per_trip := brian_fish_per_trip / brian_catch_fraction
  let brian_total := brian_trips * brian_fish_per_trip
  let chris_total := chris_trips * chris_fish_per_trip.floor
  brian_total + chris_total = 14660 := by
sorry

end total_fish_caught_l854_85466


namespace zero_in_interval_l854_85428

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x - 9

-- State the theorem
theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end zero_in_interval_l854_85428


namespace sheets_count_l854_85429

/-- The number of sheets in a set of writing materials -/
def S : ℕ := sorry

/-- The number of envelopes in a set of writing materials -/
def E : ℕ := sorry

/-- John's equation: sheets minus envelopes equals 80 -/
axiom john_equation : S - E = 80

/-- Mary's equation: sheets equals 4 times envelopes -/
axiom mary_equation : S = 4 * E

/-- Theorem: The number of sheets in each set is 320 -/
theorem sheets_count : S = 320 := by sorry

end sheets_count_l854_85429


namespace yellow_marbles_count_l854_85451

/-- Given a bowl of marbles with the following properties:
  - Total number of marbles is 19
  - Marbles are split into yellow, blue, and red
  - Ratio of blue to red marbles is 3:4
  - There are 3 more red marbles than yellow marbles
  Prove that the number of yellow marbles is 5. -/
theorem yellow_marbles_count :
  ∀ (yellow blue red : ℕ),
  yellow + blue + red = 19 →
  4 * blue = 3 * red →
  red = yellow + 3 →
  yellow = 5 := by
sorry

end yellow_marbles_count_l854_85451


namespace coronavirus_cases_in_new_york_l854_85430

theorem coronavirus_cases_in_new_york :
  ∀ (new_york california texas : ℕ),
    california = new_york / 2 →
    california = texas + 400 →
    new_york + california + texas = 3600 →
    new_york = 800 := by
  sorry

end coronavirus_cases_in_new_york_l854_85430


namespace complement_P_intersect_Q_l854_85475

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 0 < Real.log x ∧ Real.log x ≤ Real.log 2}

theorem complement_P_intersect_Q : 
  (Set.compl P) ∩ Q = Set.Ioo 1 2 := by sorry

end complement_P_intersect_Q_l854_85475


namespace pebble_difference_l854_85408

/-- Given Shawn's pebble collection and painting process, prove the difference between blue and yellow pebbles. -/
theorem pebble_difference (total : ℕ) (red : ℕ) (blue : ℕ) (groups : ℕ) : 
  total = 40 →
  red = 9 →
  blue = 13 →
  groups = 3 →
  let remaining := total - red - blue
  let yellow := remaining / groups
  blue - yellow = 7 := by
  sorry

end pebble_difference_l854_85408


namespace infinitely_many_n_for_f_congruence_l854_85471

/-- The function f(p, n) represents the largest integer k such that p^k divides n! -/
def f (p n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem infinitely_many_n_for_f_congruence 
  (p : ℕ) 
  (m c : ℕ+) 
  (h_prime : Nat.Prime p) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, f p n ≡ c.val [MOD m.val] := by sorry

end infinitely_many_n_for_f_congruence_l854_85471


namespace min_radios_sold_l854_85473

/-- Proves the minimum value of n given the radio sales problem conditions -/
theorem min_radios_sold (n d₁ : ℕ) : 
  0 < n → 
  0 < d₁ → 
  d₁ % n = 0 → 
  10 * n - 30 = 80 → 
  ∀ m : ℕ, 0 < m → 10 * m - 30 = 80 → n ≤ m :=
by sorry

end min_radios_sold_l854_85473


namespace gcf_54_81_l854_85494

theorem gcf_54_81 : Nat.gcd 54 81 = 27 := by
  sorry

end gcf_54_81_l854_85494


namespace eggs_per_box_l854_85465

/-- Given that there are 6 eggs in 2 boxes and each box contains some eggs,
    prove that the number of eggs in each box is 3. -/
theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) (eggs_per_box : ℕ) 
  (h1 : total_eggs = 6)
  (h2 : num_boxes = 2)
  (h3 : eggs_per_box * num_boxes = total_eggs)
  (h4 : eggs_per_box > 0) :
  eggs_per_box = 3 := by
  sorry

end eggs_per_box_l854_85465


namespace full_price_revenue_l854_85455

/-- Represents the concert ticket sales problem. -/
structure ConcertTickets where
  total_tickets : ℕ
  total_revenue : ℕ
  full_price : ℕ
  discounted_price : ℕ
  full_price_tickets : ℕ
  discounted_tickets : ℕ

/-- The revenue generated from full-price tickets is $4500. -/
theorem full_price_revenue (ct : ConcertTickets)
  (h1 : ct.total_tickets = 200)
  (h2 : ct.total_revenue = 4500)
  (h3 : ct.discounted_price = ct.full_price / 3)
  (h4 : ct.total_tickets = ct.full_price_tickets + ct.discounted_tickets)
  (h5 : ct.total_revenue = ct.full_price * ct.full_price_tickets + ct.discounted_price * ct.discounted_tickets) :
  ct.full_price * ct.full_price_tickets = 4500 := by
  sorry

end full_price_revenue_l854_85455


namespace rectangle_perimeter_l854_85480

theorem rectangle_perimeter (long_side : ℝ) (short_side_difference : ℝ) :
  long_side = 1 →
  short_side_difference = 2/8 →
  let short_side := long_side - short_side_difference
  2 * long_side + 2 * short_side = 3.5 := by
  sorry

end rectangle_perimeter_l854_85480


namespace cricket_team_right_handed_players_l854_85484

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 58)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + (2 * (total_players - throwers) / 3)) = 51 := by
  sorry

end cricket_team_right_handed_players_l854_85484


namespace union_of_M_and_N_l854_85402

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end union_of_M_and_N_l854_85402


namespace product_correction_l854_85464

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (a b : ℕ) :
  a ≥ 10 ∧ a < 100 →  -- a is a two-digit number
  a > 0 ∧ b > 0 →  -- a and b are positive
  (reverse_digits a) * b = 143 →
  a * b = 341 := by
sorry

end product_correction_l854_85464


namespace ferry_crossings_parity_ferry_crossings_opposite_ferry_after_99_crossings_l854_85460

/-- Represents the two banks of the river --/
inductive Bank : Type
| Left : Bank
| Right : Bank

/-- Returns the opposite bank --/
def opposite_bank (b : Bank) : Bank :=
  match b with
  | Bank.Left => Bank.Right
  | Bank.Right => Bank.Left

/-- Represents the state of the ferry after a number of crossings --/
def ferry_position (start : Bank) (crossings : Nat) : Bank :=
  if crossings % 2 = 0 then start else opposite_bank start

theorem ferry_crossings_parity (start : Bank) (crossings : Nat) :
  ferry_position start crossings = start ↔ crossings % 2 = 0 :=
sorry

theorem ferry_crossings_opposite (start : Bank) (crossings : Nat) :
  ferry_position start crossings = opposite_bank start ↔ crossings % 2 = 1 :=
sorry

theorem ferry_after_99_crossings (start : Bank) :
  ferry_position start 99 = opposite_bank start :=
sorry

end ferry_crossings_parity_ferry_crossings_opposite_ferry_after_99_crossings_l854_85460


namespace passengers_from_other_continents_l854_85483

theorem passengers_from_other_continents 
  (total : ℕ) 
  (h_total : total = 240) 
  (h_na : total / 3 = 80) 
  (h_eu : total / 8 = 30) 
  (h_af : total / 5 = 48) 
  (h_as : total / 6 = 40) : 
  total - (total / 3 + total / 8 + total / 5 + total / 6) = 42 := by
  sorry

end passengers_from_other_continents_l854_85483


namespace perpendicular_vectors_l854_85498

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is perpendicular to c, 
    then the k-coordinate of c is -3. -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) : 
  a = (Real.sqrt 3, 1) → 
  b = (0, -1) → 
  c.1 = k → 
  c.2 = Real.sqrt 3 → 
  (a.1 - 2 * b.1, a.2 - 2 * b.2) • c = 0 → 
  k = -3 := by sorry

end perpendicular_vectors_l854_85498


namespace circle_and_lines_properties_l854_85489

-- Define the circle C
def circle_C (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 4}

-- Define the tangent line
def tangent_line := {(x, y) : ℝ × ℝ | 3*x - 4*y + 4 = 0}

-- Define the intersecting line l
def line_l (k : ℝ) := {(x, y) : ℝ × ℝ | y = k*x - 3}

-- Main theorem
theorem circle_and_lines_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ (p : ℝ × ℝ), p ∈ circle_C a → p ∉ tangent_line) ∧
  (∃ (q : ℝ × ℝ), q ∈ circle_C a ∧ q ∈ tangent_line) →
  (a = 2) ∧
  (∀ (k x₁ y₁ x₂ y₂ : ℝ),
    (x₁, y₁) ∈ circle_C 2 ∧ (x₁, y₁) ∈ line_l k ∧
    (x₂, y₂) ∈ circle_C 2 ∧ (x₂, y₂) ∈ line_l k ∧
    (x₁, y₁) ≠ (x₂, y₂) →
    (k = 3 → x₁ * x₂ + y₁ * y₂ = -9/5) ∧
    (x₁ * x₂ + y₁ * y₂ = 8 → k = (-3 + Real.sqrt 29) / 4)) :=
sorry

end circle_and_lines_properties_l854_85489


namespace round_trip_percentage_is_25_percent_l854_85449

/-- Represents the percentage of ship passengers holding round-trip tickets -/
def round_trip_percentage : ℝ := 25

/-- Represents the percentage of all passengers who held round-trip tickets and took their cars aboard -/
def round_trip_with_car_percentage : ℝ := 20

/-- Represents the percentage of round-trip ticket holders who did not take their cars aboard -/
def round_trip_without_car_percentage : ℝ := 20

/-- Proves that the percentage of ship passengers holding round-trip tickets is 25% -/
theorem round_trip_percentage_is_25_percent :
  round_trip_percentage = 25 :=
by sorry

end round_trip_percentage_is_25_percent_l854_85449


namespace cycling_equation_correct_l854_85467

/-- Represents the scenario of two employees cycling to work -/
def cycling_scenario (x : ℝ) : Prop :=
  let distance : ℝ := 5000
  let speed_ratio : ℝ := 1.5
  let time_difference : ℝ := 10
  (distance / x) - (distance / (speed_ratio * x)) = time_difference

/-- Proves that the equation correctly represents the cycling scenario -/
theorem cycling_equation_correct :
  ∀ x : ℝ, x > 0 → cycling_scenario x :=
by
  sorry

end cycling_equation_correct_l854_85467


namespace minimum_red_marbles_l854_85485

theorem minimum_red_marbles (r w g : ℕ) : 
  g ≥ (2 * w) / 3 →
  g ≤ r / 4 →
  w + g ≥ 72 →
  (∀ r' : ℕ, (∃ w' g' : ℕ, g' ≥ (2 * w') / 3 ∧ g' ≤ r' / 4 ∧ w' + g' ≥ 72) → r' ≥ r) →
  r = 120 := by
sorry

end minimum_red_marbles_l854_85485


namespace rationalize_and_product_l854_85472

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2 : ℝ) + Real.sqrt 5) / ((3 : ℝ) - 2 * Real.sqrt 5) = A + B * Real.sqrt C) ∧
  A * B * C = -560 := by
  sorry

end rationalize_and_product_l854_85472


namespace boat_against_stream_distance_l854_85433

/-- Proves the distance traveled against the stream given boat speed and along-stream distance --/
theorem boat_against_stream_distance
  (boat_speed : ℝ)
  (along_stream_distance : ℝ)
  (h1 : boat_speed = 8)
  (h2 : along_stream_distance = 11)
  : (2 * boat_speed - along_stream_distance) = 5 := by
  sorry

end boat_against_stream_distance_l854_85433


namespace ten_mile_taxi_cost_l854_85463

/-- Calculates the cost of a taxi ride given the base fare, per-mile rate, and distance traveled. -/
def taxiRideCost (baseFare : ℝ) (perMileRate : ℝ) (distance : ℝ) : ℝ :=
  baseFare + perMileRate * distance

/-- Theorem stating that a 10-mile taxi ride costs $5.00 given the specified base fare and per-mile rate. -/
theorem ten_mile_taxi_cost :
  let baseFare : ℝ := 2.00
  let perMileRate : ℝ := 0.30
  let distance : ℝ := 10
  taxiRideCost baseFare perMileRate distance = 5.00 := by
  sorry


end ten_mile_taxi_cost_l854_85463


namespace cubic_equation_solutions_l854_85417

theorem cubic_equation_solutions :
  ∀ m n : ℤ, (n^3 + m^3 + 231 = n^2 * m^2 + n * m) ↔ ((m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4)) :=
by sorry

end cubic_equation_solutions_l854_85417


namespace average_weight_b_c_l854_85443

/-- Given three weights a, b, and c, prove that the average weight of b and c is 50 kg
    under the specified conditions. -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 60 →  -- The average weight of a, b, and c is 60 kg
  (a + b) / 2 = 70 →      -- The average weight of a and b is 70 kg
  b = 60 →                -- The weight of b is 60 kg
  (b + c) / 2 = 50 :=     -- The average weight of b and c is 50 kg
by sorry

end average_weight_b_c_l854_85443


namespace shelter_final_count_l854_85420

/-- Represents the number of cats in the shelter at different points in time. -/
structure CatCount where
  initial : Nat
  afterDoubling : Nat
  afterMonday : Nat
  afterTuesday : Nat
  afterWednesday : Nat
  afterThursday : Nat
  afterFriday : Nat
  afterReclaiming : Nat
  final : Nat

/-- Represents the events that occurred during the week at the animal shelter. -/
def shelterWeek (c : CatCount) : Prop :=
  c.afterDoubling = c.initial * 2 ∧
  c.afterDoubling = 48 ∧
  c.afterMonday = c.afterDoubling - 3 ∧
  c.afterTuesday = c.afterMonday + 5 ∧
  c.afterWednesday = c.afterTuesday - 3 ∧
  c.afterThursday = c.afterWednesday + 5 ∧
  c.afterFriday = c.afterThursday - 3 ∧
  c.afterReclaiming = c.afterFriday - 3 ∧
  c.final = c.afterReclaiming - 5

/-- Theorem stating that after the events of the week, the shelter has 41 cats. -/
theorem shelter_final_count (c : CatCount) :
  shelterWeek c → c.final = 41 := by
  sorry


end shelter_final_count_l854_85420


namespace tan_angle_equality_l854_85409

theorem tan_angle_equality (n : Int) :
  -90 < n ∧ n < 90 →
  Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180) →
  n = 75 := by
sorry

end tan_angle_equality_l854_85409


namespace even_function_implies_a_eq_neg_one_l854_85422

def f (x a : ℝ) : ℝ := (x + 1) * (x + a)

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = -1 := by
  sorry

end even_function_implies_a_eq_neg_one_l854_85422


namespace tangent_circle_radius_l854_85413

/-- A 30-60-90 triangle with shortest side length 2 -/
structure Triangle30_60_90 where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  is_30_60_90 : True  -- Placeholder for the triangle's angle properties
  de_length : dist D E = 2

/-- A circle tangent to coordinate axes and parts of the triangle -/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  triangle : Triangle30_60_90
  tangent_to_axes : True  -- Placeholder for tangency to coordinate axes
  tangent_to_leg : True   -- Placeholder for tangency to one leg of the triangle
  tangent_to_hypotenuse : True  -- Placeholder for tangency to hypotenuse

/-- The main theorem stating the radius of the tangent circle -/
theorem tangent_circle_radius (c : TangentCircle) : c.r = (5 + Real.sqrt 3) / 2 := by
  sorry

end tangent_circle_radius_l854_85413


namespace avery_theorem_l854_85456

/-- A shape with a certain number of 90-degree angles -/
structure Shape :=
  (angles : ℕ)

/-- A rectangular park is a shape with 4 90-degree angles -/
def rectangular_park : Shape :=
  ⟨4⟩

/-- Avery's visit to two places -/
structure AverysVisit :=
  (place1 : Shape)
  (place2 : Shape)
  (total_angles : ℕ)

/-- A rectangle or square is a shape with 4 90-degree angles -/
def rectangle_or_square (s : Shape) : Prop :=
  s.angles = 4

/-- The theorem to be proved -/
theorem avery_theorem (visit : AverysVisit) :
  visit.place1 = rectangular_park →
  visit.total_angles = 8 →
  rectangle_or_square visit.place2 :=
sorry

end avery_theorem_l854_85456


namespace min_value_quadratic_l854_85488

theorem min_value_quadratic (x : ℝ) : 
  ∃ (z_min : ℝ), ∀ (z : ℝ), z = x^2 + 16*x + 20 → z ≥ z_min ∧ ∃ (x_min : ℝ), x_min^2 + 16*x_min + 20 = z_min :=
by
  sorry

end min_value_quadratic_l854_85488


namespace somu_father_age_ratio_l854_85448

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.somu = 14 ∧
  ages.somu - 7 = (ages.father - 7) / 5

/-- The theorem to prove -/
theorem somu_father_age_ratio (ages : Ages) :
  problem_conditions ages →
  (ages.somu : ℚ) / ages.father = 1 / 3 := by
  sorry

#check somu_father_age_ratio

end somu_father_age_ratio_l854_85448


namespace saved_amount_l854_85469

theorem saved_amount (x : ℕ) : (3 * x - 42)^2 = 2241 → x = 30 := by
  sorry

end saved_amount_l854_85469


namespace convex_polygon_symmetry_l854_85411

/-- A polygon is convex if all its interior angles are less than or equal to 180 degrees. -/
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

/-- A polygon has a center of symmetry if there exists a point such that every point of the polygon has a corresponding point that is equidistant from the center but in the opposite direction. -/
def HasCenterOfSymmetry (P : Set (ℝ × ℝ)) : Prop := sorry

/-- A polygon can be divided into smaller polygons if there exists a partition of the polygon into a finite number of non-overlapping smaller polygons. -/
def CanBeDivided (P : Set (ℝ × ℝ)) (subPolygons : Finset (Set (ℝ × ℝ))) : Prop := sorry

theorem convex_polygon_symmetry 
  (P : Set (ℝ × ℝ)) 
  (subPolygons : Finset (Set (ℝ × ℝ))) 
  (h1 : ConvexPolygon P) 
  (h2 : CanBeDivided P subPolygons) 
  (h3 : ∀ Q ∈ subPolygons, HasCenterOfSymmetry Q) : 
  HasCenterOfSymmetry P := by
  sorry

end convex_polygon_symmetry_l854_85411


namespace not_divisible_by_1955_l854_85477

theorem not_divisible_by_1955 : ∀ n : ℤ, ¬(1955 ∣ (n^2 + n + 1)) := by sorry

end not_divisible_by_1955_l854_85477


namespace boxes_with_neither_l854_85415

/-- Represents the set of boxes in Christine's storage room. -/
def Boxes : Type := Unit

/-- The total number of boxes. -/
def total_boxes : ℕ := 15

/-- The number of boxes containing markers. -/
def boxes_with_markers : ℕ := 8

/-- The number of boxes containing sharpies. -/
def boxes_with_sharpies : ℕ := 5

/-- The number of boxes containing both markers and sharpies. -/
def boxes_with_both : ℕ := 4

/-- Theorem stating the number of boxes containing neither markers nor sharpies. -/
theorem boxes_with_neither (b : Boxes) :
  total_boxes - (boxes_with_markers + boxes_with_sharpies - boxes_with_both) = 6 := by
  sorry


end boxes_with_neither_l854_85415


namespace unique_positive_integers_sum_l854_85437

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 3 + 5 / 3)

theorem unique_positive_integers_sum (d e f : ℕ+) : 
  y^50 = 3*y^48 + 10*y^45 + 9*y^43 - y^25 + (d:ℝ)*y^21 + (e:ℝ)*y^19 + (f:ℝ)*y^15 →
  d + e + f = 119 := by sorry

end unique_positive_integers_sum_l854_85437


namespace kristin_green_beans_count_l854_85439

/-- Represents the number of vegetables a person has -/
structure VegetableCount where
  carrots : ℕ
  cucumbers : ℕ
  bellPeppers : ℕ
  greenBeans : ℕ

/-- The problem statement -/
theorem kristin_green_beans_count 
  (jaylen : VegetableCount)
  (kristin : VegetableCount)
  (h1 : jaylen.carrots = 5)
  (h2 : jaylen.cucumbers = 2)
  (h3 : jaylen.bellPeppers = 2 * kristin.bellPeppers)
  (h4 : jaylen.greenBeans = kristin.greenBeans / 2 - 3)
  (h5 : jaylen.carrots + jaylen.cucumbers + jaylen.bellPeppers + jaylen.greenBeans = 18)
  (h6 : kristin.bellPeppers = 2) :
  kristin.greenBeans = 20 := by
  sorry

end kristin_green_beans_count_l854_85439


namespace chipped_marbles_count_l854_85497

def marble_counts : List Nat := [17, 20, 22, 24, 26, 35, 37, 40]

def total_marbles : Nat := marble_counts.sum

theorem chipped_marbles_count (jane_count george_count : Nat) 
  (h1 : jane_count = 3 * george_count)
  (h2 : jane_count + george_count = total_marbles - (marble_counts.get! 0 + marble_counts.get! 7))
  (h3 : ∃ (i j : Fin 8), i ≠ j ∧ 
    marble_counts.get! i.val + marble_counts.get! j.val = total_marbles - (jane_count + george_count) ∧
    (marble_counts.get! i.val = 40 ∨ marble_counts.get! j.val = 40)) :
  40 ∈ marble_counts ∧ 
  ∃ (i j : Fin 8), i ≠ j ∧ 
    marble_counts.get! i.val + marble_counts.get! j.val = total_marbles - (jane_count + george_count) ∧
    (marble_counts.get! i.val = 40 ∨ marble_counts.get! j.val = 40) :=
by sorry

end chipped_marbles_count_l854_85497


namespace fraction_subtraction_simplification_l854_85419

theorem fraction_subtraction_simplification :
  (8 : ℚ) / 29 - (5 : ℚ) / 87 = (19 : ℚ) / 87 ∧ 
  (∀ n d : ℤ, n ≠ 0 → (19 : ℚ) / 87 = (n : ℚ) / d → (abs n = 19 ∧ abs d = 87)) :=
by sorry

end fraction_subtraction_simplification_l854_85419


namespace smallest_solution_congruence_l854_85441

theorem smallest_solution_congruence :
  ∃! x : ℕ+, (5 * x.val ≡ 14 [ZMOD 26]) ∧
    ∀ y : ℕ+, (5 * y.val ≡ 14 [ZMOD 26]) → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_solution_congruence_l854_85441


namespace strict_manager_proposal_l854_85452

/-- Represents the total monthly salary before changes -/
def initial_total_salary : ℕ := 10000

/-- Represents the total monthly salary after the kind manager's proposal -/
def kind_manager_total_salary : ℕ := 24000

/-- Represents the salary threshold -/
def salary_threshold : ℕ := 500

/-- Represents the number of employees -/
def total_employees : ℕ := 14

theorem strict_manager_proposal (x y : ℕ) 
  (h1 : x + y = total_employees)
  (h2 : 500 * x + y * salary_threshold ≤ initial_total_salary)
  (h3 : 3 * 500 * x + (initial_total_salary - 500 * x) + 1000 * y = kind_manager_total_salary) :
  500 * (x + y) = 7000 := by
  sorry

end strict_manager_proposal_l854_85452


namespace dish_price_proof_l854_85423

theorem dish_price_proof (discount_rate : Real) (tip_rate : Real) (price_difference : Real) :
  let original_price : Real := 36
  let john_payment := original_price * (1 - discount_rate) + original_price * tip_rate
  let jane_payment := original_price * (1 - discount_rate) * (1 + tip_rate)
  discount_rate = 0.1 ∧ tip_rate = 0.15 ∧ price_difference = 0.54 →
  john_payment - jane_payment = price_difference :=
by
  sorry

end dish_price_proof_l854_85423


namespace average_age_combined_l854_85486

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 40 →
  n_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents) = 28.8 := by
sorry

end average_age_combined_l854_85486


namespace equal_weekend_days_count_l854_85490

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Checks if starting the month on a given day results in equal Saturdays and Sundays -/
def equalWeekendDays (startDay : DayOfWeek) : Bool :=
  sorry

/-- Counts the number of days that result in equal Saturdays and Sundays when used as the start day -/
def countEqualWeekendDays : Nat :=
  sorry

theorem equal_weekend_days_count :
  countEqualWeekendDays = 2 :=
sorry

end equal_weekend_days_count_l854_85490


namespace election_votes_l854_85457

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) : 
  total_votes = 9000 →
  invalid_percent = 30 / 100 →
  winner_percent = 60 / 100 →
  (total_votes : ℚ) * (1 - invalid_percent) * (1 - winner_percent) = 2520 := by
sorry

end election_votes_l854_85457


namespace not_multiple_of_five_l854_85412

theorem not_multiple_of_five : ∃ n : ℕ, (2015^2 / 5^2 = n) ∧ ¬(∃ k : ℕ, n = 5 * k) ∧
  (∃ k₁ : ℕ, 2019^2 - 2014^2 = 5 * k₁) ∧
  (∃ k₂ : ℕ, 2019^2 * 10^2 = 5 * k₂) ∧
  (∃ k₃ : ℕ, 2020^2 / 101^2 = 5 * k₃) ∧
  (∃ k₄ : ℕ, 2010^2 - 2005^2 = 5 * k₄) :=
by sorry

#check not_multiple_of_five

end not_multiple_of_five_l854_85412


namespace max_g_value_l854_85405

theorem max_g_value (t : Real) (h : t ∈ Set.Icc 0 Real.pi) : 
  let g := fun (t : Real) => (4 * Real.cos t + 5) * (1 - Real.cos t)^2
  ∃ (max_val : Real), max_val = 27/4 ∧ 
    (∀ s, s ∈ Set.Icc 0 Real.pi → g s ≤ max_val) ∧
    g (2 * Real.pi / 3) = max_val :=
by sorry

end max_g_value_l854_85405


namespace mixture_ratio_after_mixing_l854_85406

/-- Represents a mixture of two liquids -/
structure Mixture where
  total : ℚ
  ratio_alpha : ℕ
  ratio_beta : ℕ

/-- Calculates the amount of alpha in a mixture -/
def alpha_amount (m : Mixture) : ℚ :=
  m.total * (m.ratio_alpha : ℚ) / ((m.ratio_alpha + m.ratio_beta) : ℚ)

/-- Calculates the amount of beta in a mixture -/
def beta_amount (m : Mixture) : ℚ :=
  m.total * (m.ratio_beta : ℚ) / ((m.ratio_alpha + m.ratio_beta) : ℚ)

theorem mixture_ratio_after_mixing (m1 m2 : Mixture)
  (h1 : m1.total = 6 ∧ m1.ratio_alpha = 7 ∧ m1.ratio_beta = 2)
  (h2 : m2.total = 9 ∧ m2.ratio_alpha = 4 ∧ m2.ratio_beta = 7) :
  (alpha_amount m1 + alpha_amount m2) / (beta_amount m1 + beta_amount m2) = 262 / 233 := by
  sorry

#eval 262 / 233

end mixture_ratio_after_mixing_l854_85406


namespace min_p_for_quadratic_roots_in_unit_interval_l854_85432

theorem min_p_for_quadratic_roots_in_unit_interval :
  (∃ (p : ℕ+),
    (∀ p' : ℕ+, p' < p →
      ¬∃ (q r : ℕ+),
        (∃ (x y : ℝ),
          0 < x ∧ x < 1 ∧
          0 < y ∧ y < 1 ∧
          x ≠ y ∧
          p' * x^2 - q * x + r = 0 ∧
          p' * y^2 - q * y + r = 0)) ∧
    (∃ (q r : ℕ+),
      ∃ (x y : ℝ),
        0 < x ∧ x < 1 ∧
        0 < y ∧ y < 1 ∧
        x ≠ y ∧
        p * x^2 - q * x + r = 0 ∧
        p * y^2 - q * y + r = 0)) ∧
  (∀ p : ℕ+,
    (∀ p' : ℕ+, p' < p →
      ¬∃ (q r : ℕ+),
        (∃ (x y : ℝ),
          0 < x ∧ x < 1 ∧
          0 < y ∧ y < 1 ∧
          x ≠ y ∧
          p' * x^2 - q * x + r = 0 ∧
          p' * y^2 - q * y + r = 0)) ∧
    (∃ (q r : ℕ+),
      ∃ (x y : ℝ),
        0 < x ∧ x < 1 ∧
        0 < y ∧ y < 1 ∧
        x ≠ y ∧
        p * x^2 - q * x + r = 0 ∧
        p * y^2 - q * y + r = 0) →
    p = 5) :=
by sorry

end min_p_for_quadratic_roots_in_unit_interval_l854_85432


namespace unique_two_digit_sum_diff_product_l854_85468

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_sum_diff_product (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    n = 10 * x + y ∧
    1 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    n = (x + y) * (y - x)

theorem unique_two_digit_sum_diff_product :
  ∃! n : ℕ, is_two_digit n ∧ digits_sum_diff_product n ∧ n = 48 := by
  sorry

end unique_two_digit_sum_diff_product_l854_85468


namespace difference_of_squares_l854_85435

theorem difference_of_squares (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 := by
  sorry

end difference_of_squares_l854_85435


namespace positive_interval_l854_85453

theorem positive_interval (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end positive_interval_l854_85453


namespace exactly_one_hit_probability_l854_85431

def probability_exactly_one_hit (p_a p_b : ℝ) : ℝ :=
  p_a * (1 - p_b) + (1 - p_a) * p_b

theorem exactly_one_hit_probability :
  let p_a : ℝ := 1/2
  let p_b : ℝ := 1/3
  probability_exactly_one_hit p_a p_b = 1/2 := by
  sorry

end exactly_one_hit_probability_l854_85431


namespace susan_homework_time_l854_85491

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem susan_homework_time : 
  let homeworkStart : Time := { hours := 13, minutes := 59 }
  let homeworkDuration : Nat := 96
  let practiceStart : Time := { hours := 16, minutes := 0 }
  let homeworkEnd := addMinutes homeworkStart homeworkDuration
  timeDifference homeworkEnd practiceStart = 25 := by
  sorry

end susan_homework_time_l854_85491


namespace lines_cannot_form_triangle_l854_85476

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  let x := (l1.b * l3.c - l3.b * l1.c) / (l1.a * l3.b - l3.a * l1.b)
  let y := (l3.a * l1.c - l1.a * l3.c) / (l1.a * l3.b - l3.a * l1.b)
  l2.a * x + l2.b * y = l2.c

/-- The main theorem -/
theorem lines_cannot_form_triangle (m : ℝ) : 
  let l1 : Line := ⟨4, 1, 4⟩
  let l2 : Line := ⟨m, 1, 0⟩
  let l3 : Line := ⟨2, -3, 4⟩
  (parallel l1 l2 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) →
  m = 4 ∨ m = 1/2 ∨ m = -2/3 :=
by sorry


end lines_cannot_form_triangle_l854_85476
