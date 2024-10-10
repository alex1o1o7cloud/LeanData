import Mathlib

namespace largest_movable_n_l738_73860

/-- Represents the rules for moving cards between boxes -/
structure CardMoveRules where
  /-- A card can be placed in an empty box -/
  place_in_empty : Bool
  /-- A card can be placed on top of a card with a number one greater than its own -/
  place_on_greater : Bool

/-- Represents the configuration of card boxes -/
structure BoxConfiguration where
  /-- Number of blue boxes -/
  k : Nat
  /-- Number of cards (2n) -/
  card_count : Nat
  /-- Rules for moving cards -/
  move_rules : CardMoveRules

/-- Determines if all cards can be moved to blue boxes given a configuration -/
def can_move_all_cards (config : BoxConfiguration) : Prop :=
  ∃ (final_state : List (List Nat)), 
    final_state.length = config.k ∧ 
    final_state.all (λ box => box.length > 0) ∧
    final_state.join.toFinset = Finset.range config.card_count

/-- The main theorem stating the largest possible n for which all cards can be moved -/
theorem largest_movable_n (k : Nat) (h : k > 1) :
  ∀ n : Nat, (
    let config := BoxConfiguration.mk k (2 * n) 
      { place_in_empty := true, place_on_greater := true }
    can_move_all_cards config ↔ n ≤ k - 1
  ) := by sorry

end largest_movable_n_l738_73860


namespace inequality_proof_l738_73822

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  a / (b^2 * (c + 1)) + b / (c^2 * (a + 1)) + c / (a^2 * (b + 1)) ≥ 3/2 := by
  sorry

end inequality_proof_l738_73822


namespace quadratic_solution_range_l738_73890

/-- The quadratic equation x^2 + (m-1)x + 1 = 0 has solutions in [0,2] if and only if m < -1 -/
theorem quadratic_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 + (m-1)*x + 1 = 0) ↔ m < -1 := by
  sorry

end quadratic_solution_range_l738_73890


namespace quadratic_to_vertex_form_l738_73839

theorem quadratic_to_vertex_form (x : ℝ) : 
  x^2 - 4*x + 5 = (x - 2)^2 + 1 := by sorry

end quadratic_to_vertex_form_l738_73839


namespace fifteenth_student_age_l738_73886

theorem fifteenth_student_age
  (n : ℕ)
  (total_students : n = 15)
  (avg_age : ℝ)
  (total_avg : avg_age = 15)
  (group1_size group2_size : ℕ)
  (group1_avg group2_avg : ℝ)
  (group_sizes : group1_size = 7 ∧ group2_size = 7)
  (group_avgs : group1_avg = 14 ∧ group2_avg = 16)
  : ∃ (fifteenth_age : ℝ), fifteenth_age = 15 :=
by
  sorry

end fifteenth_student_age_l738_73886


namespace profit_maximized_at_25_l738_73872

/-- Profit function for the commodity -/
def profit (x : ℤ) : ℤ := (x - 20) * (30 - x)

/-- Theorem stating that profit is maximized at x = 25 -/
theorem profit_maximized_at_25 :
  ∀ x : ℤ, 20 ≤ x → x ≤ 30 → profit x ≤ profit 25 :=
by
  sorry

#check profit_maximized_at_25

end profit_maximized_at_25_l738_73872


namespace vector_equation_l738_73874

theorem vector_equation (a b c : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → c = (-1, -2) → 
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b :=
by sorry

end vector_equation_l738_73874


namespace roots_sum_abs_l738_73846

theorem roots_sum_abs (d e f n : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = (x - d) * (x - e) * (x - f)) →
  abs d + abs e + abs f = 98 := by
sorry

end roots_sum_abs_l738_73846


namespace quadratic_expansion_l738_73873

theorem quadratic_expansion (m n : ℝ) :
  (∀ x : ℝ, (x + 4) * (x - 2) = x^2 + m*x + n) →
  m = 2 ∧ n = -8 := by
sorry

end quadratic_expansion_l738_73873


namespace dog_cord_length_l738_73843

theorem dog_cord_length (diameter : ℝ) (h : diameter = 30) : 
  diameter / 2 = 15 := by
  sorry

end dog_cord_length_l738_73843


namespace exists_ten_digit_number_with_composite_subnumbers_l738_73867

/-- A ten-digit number composed of ten different digits. -/
def TenDigitNumber := Fin 10 → Fin 10

/-- Checks if a number is composite. -/
def IsComposite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Checks if a four-digit number is composite. -/
def IsFourDigitComposite (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000 ∧ IsComposite n

/-- Generates all four-digit numbers from a ten-digit number by removing six digits. -/
def FourDigitSubnumbers (n : TenDigitNumber) : Set ℕ :=
  {m | ∃ (i j k l : Fin 10), i < j ∧ j < k ∧ k < l ∧
    m = n i * 1000 + n j * 100 + n k * 10 + n l}

/-- The main theorem stating that there exists a ten-digit number with the required property. -/
theorem exists_ten_digit_number_with_composite_subnumbers :
  ∃ (n : TenDigitNumber),
    (∀ i j, i ≠ j → n i ≠ n j) ∧
    (∀ m ∈ FourDigitSubnumbers n, IsFourDigitComposite m) := by
  sorry

end exists_ten_digit_number_with_composite_subnumbers_l738_73867


namespace teresa_jog_distance_l738_73835

/-- Given a speed of 5 km/h and a time of 5 hours, prove that the distance traveled is 25 km. -/
theorem teresa_jog_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 5)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 25 := by
sorry

end teresa_jog_distance_l738_73835


namespace chord_relations_l738_73875

/-- Represents a chord in a unit circle -/
structure Chord where
  length : ℝ

/-- Represents the configuration of chords in the unit circle -/
structure CircleChords where
  MP : Chord
  PQ : Chord
  NR : Chord
  MN : Chord

/-- The given configuration of chords satisfying the problem conditions -/
def given_chords : CircleChords :=
  { MP := ⟨1⟩
  , PQ := ⟨1⟩
  , NR := ⟨2⟩
  , MN := ⟨3⟩ }

theorem chord_relations (c : CircleChords) (h : c = given_chords) :
  (c.MN.length - c.NR.length = 1) ∧
  (c.MN.length * c.NR.length = 6) ∧
  (c.MN.length ^ 2 - c.NR.length ^ 2 = 5) := by
  sorry

end chord_relations_l738_73875


namespace streetlight_problem_l738_73888

/-- The number of streetlights -/
def total_streetlights : ℕ := 12

/-- The number of streetlights that need to be turned off -/
def lights_to_turn_off : ℕ := 4

/-- The number of available positions to turn off lights, considering the constraints -/
def available_positions : ℕ := total_streetlights - 5

/-- The number of ways to choose 4 non-adjacent positions from 7 available positions -/
def ways_to_turn_off_lights : ℕ := Nat.choose available_positions lights_to_turn_off

theorem streetlight_problem :
  ways_to_turn_off_lights = 35 :=
sorry

end streetlight_problem_l738_73888


namespace f_divisible_by_factorial_l738_73877

def f : ℕ → ℕ → ℕ
  | 0, 0 => 1
  | 0, _ => 0
  | _, 0 => 0
  | n+1, k+1 => (n+1) * (f (n+1) k + f n k)

theorem f_divisible_by_factorial (n k : ℕ) : 
  ∃ m : ℤ, f n k = n! * m := by sorry

end f_divisible_by_factorial_l738_73877


namespace arithmetic_sequence_sum_specific_l738_73891

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-3) 7 6 = 87 := by sorry

end arithmetic_sequence_sum_specific_l738_73891


namespace trader_loss_percentage_l738_73856

theorem trader_loss_percentage (cost_price : ℝ) (cost_price_pos : cost_price > 0) : 
  let marked_price := cost_price * 1.1
  let selling_price := marked_price * 0.9
  let loss := cost_price - selling_price
  loss / cost_price = 0.01 := by
sorry

end trader_loss_percentage_l738_73856


namespace smallest_sum_of_reciprocals_l738_73851

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1/12) :
  (∀ a b : ℕ+, a ≠ b → (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1/12 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) ∧
  (x : ℕ) + (y : ℕ) = 50 :=
sorry

end smallest_sum_of_reciprocals_l738_73851


namespace intersection_sum_l738_73836

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4
def g (x y : ℝ) : Prop := x + 5*y = 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 5 ∧
    y₁ + y₂ + y₃ = 2 :=
  sorry

end intersection_sum_l738_73836


namespace parabola_focus_l738_73806

/-- A parabola is defined by its equation relating x and y coordinates -/
structure Parabola where
  equation : ℝ → ℝ

/-- The focus of a parabola is a point (x, y) -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Predicate to check if a given point is the focus of a parabola -/
def is_focus (p : Parabola) (f : Focus) : Prop :=
  ∀ (y : ℝ), 
    let x := p.equation y
    (x - f.x)^2 + y^2 = (x - (f.x - 3))^2

/-- Theorem stating that (-3, 0) is the focus of the parabola x = -1/12 * y^2 -/
theorem parabola_focus :
  let p : Parabola := ⟨λ y => -1/12 * y^2⟩
  let f : Focus := ⟨-3, 0⟩
  is_focus p f := by
  sorry

end parabola_focus_l738_73806


namespace x_sixth_plus_inverse_l738_73821

theorem x_sixth_plus_inverse (x : ℝ) (h : x + 1/x = 7) : x^6 + 1/x^6 = 103682 := by
  sorry

end x_sixth_plus_inverse_l738_73821


namespace pass_in_later_rounds_l738_73837

/-- Represents the probability of correctly answering each question -/
structure QuestionProbabilities where
  A : ℚ
  B : ℚ
  C : ℚ

/-- Represents the interview process -/
def Interview (probs : QuestionProbabilities) : Prop :=
  probs.A = 1/2 ∧ probs.B = 1/3 ∧ probs.C = 1/4

/-- The probability of passing the interview in the second or third round -/
def PassInLaterRounds (probs : QuestionProbabilities) : ℚ :=
  7/18

/-- Theorem stating the probability of passing in later rounds -/
theorem pass_in_later_rounds (probs : QuestionProbabilities) 
  (h : Interview probs) : 
  PassInLaterRounds probs = 7/18 := by
  sorry


end pass_in_later_rounds_l738_73837


namespace chess_tournament_games_l738_73808

theorem chess_tournament_games (n : Nat) (h : n = 5) : 
  n * (n - 1) / 2 = 10 := by
  sorry

end chess_tournament_games_l738_73808


namespace circle_radius_l738_73871

/-- The radius of the circle defined by x^2 + y^2 + 2x + 6y = 0 is √10 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 2*x + 6*y = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 10 :=
by
  sorry

end circle_radius_l738_73871


namespace power_multiplication_l738_73830

theorem power_multiplication (a : ℝ) : a^3 * a^3 = a^6 := by
  sorry

end power_multiplication_l738_73830


namespace quadratic_max_value_l738_73825

/-- The maximum value of the quadratic function f(x) = -2x^2 + 4x - 18 is -16 -/
theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 4 * x - 18
  ∃ M : ℝ, M = -16 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end quadratic_max_value_l738_73825


namespace least_five_digit_square_cube_l738_73820

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n = 15625) ∧ 
  (∀ m : ℕ, m < n → m < 10000 ∨ m > 99999 ∨ ¬∃ a : ℕ, m = a^2 ∨ ¬∃ b : ℕ, m = b^3) ∧
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) :=
by sorry

end least_five_digit_square_cube_l738_73820


namespace sales_volume_correct_profit_10000_prices_max_profit_under_constraints_l738_73819

/-- Toy sales model -/
structure ToySalesModel where
  purchase_price : ℝ
  initial_price : ℝ
  initial_volume : ℝ
  price_sensitivity : ℝ
  min_price : ℝ
  min_volume : ℝ

/-- Given toy sales model -/
def given_model : ToySalesModel :=
  { purchase_price := 30
  , initial_price := 40
  , initial_volume := 600
  , price_sensitivity := 10
  , min_price := 44
  , min_volume := 540 }

/-- Sales volume as a function of price -/
def sales_volume (model : ToySalesModel) (x : ℝ) : ℝ :=
  model.initial_volume - model.price_sensitivity * (x - model.initial_price)

/-- Profit as a function of price -/
def profit (model : ToySalesModel) (x : ℝ) : ℝ :=
  (x - model.purchase_price) * (sales_volume model x)

/-- Theorem stating the correctness of the sales volume function -/
theorem sales_volume_correct (x : ℝ) (h : x > given_model.initial_price) :
  sales_volume given_model x = 1000 - 10 * x := by sorry

/-- Theorem stating the selling prices for a profit of 10,000 yuan -/
theorem profit_10000_prices :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit given_model x₁ = 10000 ∧ profit given_model x₂ = 10000 ∧
  (x₁ = 50 ∨ x₁ = 80) ∧ (x₂ = 50 ∨ x₂ = 80) := by sorry

/-- Theorem stating the maximum profit under constraints -/
theorem max_profit_under_constraints :
  ∃ x : ℝ, x ≥ given_model.min_price ∧ 
    sales_volume given_model x ≥ given_model.min_volume ∧
    ∀ y : ℝ, y ≥ given_model.min_price → 
      sales_volume given_model y ≥ given_model.min_volume →
      profit given_model x ≥ profit given_model y ∧
      profit given_model x = 8640 := by sorry

end sales_volume_correct_profit_10000_prices_max_profit_under_constraints_l738_73819


namespace stadium_seats_l738_73879

/-- Represents the number of seats in the little league stadium -/
def total_seats : ℕ := sorry

/-- Represents the number of people who came to the game -/
def people_at_game : ℕ := 47

/-- Represents the number of people holding banners -/
def people_with_banners : ℕ := 38

/-- Represents the number of empty seats -/
def empty_seats : ℕ := 45

/-- Theorem stating that the total number of seats is equal to the sum of people at the game and empty seats -/
theorem stadium_seats : total_seats = people_at_game + empty_seats := by sorry

end stadium_seats_l738_73879


namespace pure_imaginary_complex_number_l738_73838

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I).im ≠ 0 ∧ 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I).re = 0 → 
  x = 2 := by
  sorry

end pure_imaginary_complex_number_l738_73838


namespace relationship_abc_l738_73801

theorem relationship_abc (a b c : ℝ) 
  (eq1 : b + c = 6 - 4*a + 3*a^2)
  (eq2 : c - b = 4 - 4*a + a^2) : 
  a < b ∧ b ≤ c := by
  sorry

end relationship_abc_l738_73801


namespace rectangular_prism_diagonal_rectangular_prism_diagonal_h12_l738_73854

/-- Theorem: Diagonal of a rectangular prism with specific dimensions --/
theorem rectangular_prism_diagonal (h : ℝ) (l : ℝ) (w : ℝ) : 
  h = 12 → l = 2 * h → w = l / 2 → 
  Real.sqrt (l^2 + w^2 + h^2) = 12 * Real.sqrt 6 := by
  sorry

/-- Corollary: Specific case with h = 12 --/
theorem rectangular_prism_diagonal_h12 : 
  ∃ (h l w : ℝ), h = 12 ∧ l = 2 * h ∧ w = l / 2 ∧ 
  Real.sqrt (l^2 + w^2 + h^2) = 12 * Real.sqrt 6 := by
  sorry

end rectangular_prism_diagonal_rectangular_prism_diagonal_h12_l738_73854


namespace ellipse_constants_l738_73862

/-- An ellipse with given foci and a point on its curve -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  point : ℝ × ℝ

/-- The standard form constants of an ellipse -/
structure EllipseConstants where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Theorem: For an ellipse with foci at (1, 3) and (1, 7) passing through (12, 0),
    the constants in the standard form equation are as given -/
theorem ellipse_constants (e : Ellipse) 
    (h_focus1 : e.focus1 = (1, 3))
    (h_focus2 : e.focus2 = (1, 7))
    (h_point : e.point = (12, 0)) :
    ∃ (c : EllipseConstants), 
      c.a = (Real.sqrt 130 + Real.sqrt 170) / 2 ∧
      c.b = Real.sqrt (((Real.sqrt 130 + Real.sqrt 170) / 2)^2 - 4^2) ∧
      c.h = 1 ∧
      c.k = 5 ∧
      c.a > 0 ∧
      c.b > 0 ∧
      (e.point.1 - c.h)^2 / c.a^2 + (e.point.2 - c.k)^2 / c.b^2 = 1 := by
  sorry

end ellipse_constants_l738_73862


namespace xiaoning_score_is_87_l738_73899

/-- The maximum score for a student's semester physical education comprehensive score. -/
def max_score : ℝ := 100

/-- The weight of the midterm exam score in the comprehensive score calculation. -/
def midterm_weight : ℝ := 0.3

/-- The weight of the final exam score in the comprehensive score calculation. -/
def final_weight : ℝ := 0.7

/-- Xiaoning's midterm exam score as a percentage. -/
def xiaoning_midterm : ℝ := 80

/-- Xiaoning's final exam score as a percentage. -/
def xiaoning_final : ℝ := 90

/-- Calculates the comprehensive score based on midterm and final exam scores and their weights. -/
def comprehensive_score (midterm : ℝ) (final : ℝ) : ℝ :=
  midterm * midterm_weight + final * final_weight

/-- Theorem stating that Xiaoning's physical education comprehensive score is 87 points. -/
theorem xiaoning_score_is_87 :
  comprehensive_score xiaoning_midterm xiaoning_final = 87 := by
  sorry

end xiaoning_score_is_87_l738_73899


namespace ratio_problem_l738_73876

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 := by
  sorry

end ratio_problem_l738_73876


namespace total_spent_l738_73855

def lunch_cost : ℝ := 60.50
def tip_percentage : ℝ := 20

theorem total_spent (lunch_cost : ℝ) (tip_percentage : ℝ) : 
  lunch_cost * (1 + tip_percentage / 100) = 72.60 :=
by sorry

end total_spent_l738_73855


namespace angle_b_is_sixty_degrees_triangle_is_equilateral_l738_73828

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  area_formula : S = (1/2) * a * c * Real.sin B

-- Theorem 1
theorem angle_b_is_sixty_degrees (t : Triangle) 
  (h : t.a^2 + t.c^2 = t.b^2 + t.a * t.c) : 
  t.B = π/3 := by sorry

-- Theorem 2
theorem triangle_is_equilateral (t : Triangle)
  (h1 : t.a^2 + t.c^2 = t.b^2 + t.a * t.c)
  (h2 : t.b = 2)
  (h3 : t.S = Real.sqrt 3) :
  t.a = t.b ∧ t.b = t.c := by sorry

end angle_b_is_sixty_degrees_triangle_is_equilateral_l738_73828


namespace subcommittee_formation_count_l738_73850

def total_republicans : Nat := 10
def total_democrats : Nat := 8
def subcommittee_republicans : Nat := 4
def subcommittee_democrats : Nat := 3
def senior_democrat : Nat := 1

def ways_to_form_subcommittee : Nat :=
  Nat.choose total_republicans subcommittee_republicans *
  Nat.choose (total_democrats - senior_democrat) (subcommittee_democrats - senior_democrat)

theorem subcommittee_formation_count :
  ways_to_form_subcommittee = 4410 := by
  sorry

end subcommittee_formation_count_l738_73850


namespace tv_show_average_episodes_l738_73834

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

end tv_show_average_episodes_l738_73834


namespace last_amoeba_is_B_l738_73823

/-- Represents the type of a Martian amoeba -/
inductive AmoebType
  | A
  | B
  | C

/-- Represents the state of the amoeba population -/
structure AmoebState where
  countA : ℕ
  countB : ℕ
  countC : ℕ

/-- Defines the initial state of amoebas -/
def initialState : AmoebState :=
  { countA := 20, countB := 21, countC := 22 }

/-- Defines the merger rule for amoebas -/
def merge (a b : AmoebType) : AmoebType :=
  match a, b with
  | AmoebType.A, AmoebType.B => AmoebType.C
  | AmoebType.B, AmoebType.C => AmoebType.A
  | AmoebType.C, AmoebType.A => AmoebType.B
  | _, _ => a  -- This case should not occur in valid mergers

/-- Theorem: The last remaining amoeba is of type B -/
theorem last_amoeba_is_B (final : AmoebState) 
    (h_final : final.countA + final.countB + final.countC = 1) :
    ∃ (n : ℕ), n > 0 ∧ final = { countA := 0, countB := n, countC := 0 } :=
  sorry

#check last_amoeba_is_B

end last_amoeba_is_B_l738_73823


namespace prob_not_snowing_l738_73893

theorem prob_not_snowing (p_snow : ℚ) (h : p_snow = 1/4) : 1 - p_snow = 3/4 := by
  sorry

end prob_not_snowing_l738_73893


namespace lcm_18_24_l738_73885

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l738_73885


namespace g_equality_l738_73852

-- Define the function g
def g : ℝ → ℝ := fun x ↦ -4 * x^4 + 2 * x^3 - 5 * x^2 + x + 4

-- State the theorem
theorem g_equality (x : ℝ) : 4 * x^4 + 2 * x^2 - x + g x = 2 * x^3 - 3 * x^2 + 4 := by
  sorry

end g_equality_l738_73852


namespace arithmetic_sequence_remainder_l738_73897

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sequence_sum (a₁ : ℤ) (aₙ : ℤ) (n : ℕ) : ℤ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_remainder (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 3 ∧ aₙ = 347 ∧ d = 8 →
  (sequence_sum a₁ aₙ n) % 8 = 4 := by
sorry

end arithmetic_sequence_remainder_l738_73897


namespace precision_improves_with_sample_size_l738_73816

/-- A structure representing a statistical sample -/
structure Sample (α : Type*) where
  data : List α
  size : Nat

/-- A measure of precision for an estimate -/
def precision (α : Type*) : Sample α → ℝ := sorry

/-- Theorem: As sample size increases, precision improves -/
theorem precision_improves_with_sample_size (α : Type*) :
  ∀ (s1 s2 : Sample α), s1.size < s2.size → precision α s1 < precision α s2 :=
sorry

end precision_improves_with_sample_size_l738_73816


namespace sarah_trucks_l738_73815

-- Define the initial number of trucks Sarah had
def initial_trucks : ℕ := 51

-- Define the number of trucks Sarah gave away
def trucks_given_away : ℕ := 13

-- Define the number of trucks Sarah has left
def trucks_left : ℕ := 38

-- Theorem statement
theorem sarah_trucks : 
  initial_trucks = trucks_given_away + trucks_left :=
by sorry

end sarah_trucks_l738_73815


namespace race_head_start_l738_73847

theorem race_head_start (Va Vb L H : ℝ) :
  Va = (22 / 19) * Vb →
  L / Va = (L - H) / Vb →
  H = (3 / 22) * L :=
by sorry

end race_head_start_l738_73847


namespace quadratic_roots_l738_73892

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end quadratic_roots_l738_73892


namespace polynomial_factorization_l738_73805

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 7) * (x^2 + 6*x + 8) := by
sorry

end polynomial_factorization_l738_73805


namespace quadratic_minimum_l738_73895

/-- Given a quadratic function y = 2x^2 + px + q, 
    prove that q = 10 + p^2/8 when the minimum value of y is 10 -/
theorem quadratic_minimum (p : ℝ) :
  ∃ (q : ℝ), (∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) ∧
             (∃ x₀ : ℝ, 2 * x₀^2 + p * x₀ + q = 10) →
  q = 10 + p^2 / 8 :=
sorry

end quadratic_minimum_l738_73895


namespace no_2013_numbers_exist_l738_73849

theorem no_2013_numbers_exist : ¬ ∃ (S : Finset ℕ), 
  (S.card = 2013) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y) ∧
  (∀ a ∈ S, (S.sum id - a) ≥ a^2) :=
by sorry

end no_2013_numbers_exist_l738_73849


namespace max_table_sum_l738_73883

def numbers : List ℕ := [3, 5, 7, 11, 17, 19]

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ 
  d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  (a = b ∧ b = c) ∨ (d = e ∧ e = f)

def table_sum (a b c d e f : ℕ) : ℕ :=
  a*d + a*e + a*f + b*d + b*e + b*f + c*d + c*e + c*f

theorem max_table_sum :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    table_sum a b c d e f ≤ 1995 ∧
    (∃ a b c d e f : ℕ, 
      is_valid_arrangement a b c d e f ∧ 
      table_sum a b c d e f = 1995 ∧
      (a = 19 ∧ b = 19 ∧ c = 19) ∨ (d = 19 ∧ e = 19 ∧ f = 19)) := by
  sorry

end max_table_sum_l738_73883


namespace annie_passes_bonnie_at_six_laps_l738_73864

/-- Represents the track and runners' properties -/
structure RaceSetup where
  trackLength : ℝ
  annieSpeedFactor : ℝ
  bonnieAcceleration : ℝ

/-- Calculates the number of laps Annie runs when she first passes Bonnie -/
def lapsWhenAnniePasses (setup : RaceSetup) (bonnieInitialSpeed : ℝ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem annie_passes_bonnie_at_six_laps (setup : RaceSetup) (bonnieInitialSpeed : ℝ) 
    (h1 : setup.trackLength = 300)
    (h2 : setup.annieSpeedFactor = 1.2)
    (h3 : setup.bonnieAcceleration = 0.1) :
  lapsWhenAnniePasses setup bonnieInitialSpeed = 6 := by
  sorry

end annie_passes_bonnie_at_six_laps_l738_73864


namespace bananas_in_jar_l738_73824

/-- The number of bananas originally in the jar -/
def original_bananas : ℕ := 46

/-- The number of bananas removed from the jar -/
def removed_bananas : ℕ := 5

/-- The number of bananas left in the jar after removal -/
def remaining_bananas : ℕ := 41

/-- Theorem stating that the original number of bananas is equal to the sum of removed and remaining bananas -/
theorem bananas_in_jar : original_bananas = removed_bananas + remaining_bananas := by
  sorry

end bananas_in_jar_l738_73824


namespace arithmetic_sequence_eighth_term_l738_73884

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth_term : a 5 = 10)
  (h_sum_first_three : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 :=
sorry

end arithmetic_sequence_eighth_term_l738_73884


namespace book_selling_price_l738_73882

/-- Calculates the selling price of an item given its cost price and profit rate -/
def selling_price (cost_price : ℚ) (profit_rate : ℚ) : ℚ :=
  cost_price * (1 + profit_rate)

/-- Theorem: The selling price of a book with cost price Rs 50 and profit rate 40% is Rs 70 -/
theorem book_selling_price :
  selling_price 50 (40 / 100) = 70 := by
  sorry

end book_selling_price_l738_73882


namespace ellipse_b_value_l738_73861

/-- Define an ellipse with foci F1 and F2, and a point P on the ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h1 : a > b
  h2 : b > 0
  h3 : P.1^2 / a^2 + P.2^2 / b^2 = 1  -- P is on the ellipse

/-- The dot product of PF1 and PF2 is zero -/
def orthogonal_foci (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

/-- The area of triangle PF1F2 is 9 -/
def triangle_area (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  abs (PF1.1 * PF2.2 - PF1.2 * PF2.1) / 2 = 9

/-- Main theorem: If the foci are orthogonal from P and the triangle area is 9, then b = 3 -/
theorem ellipse_b_value (e : Ellipse) 
  (h_orth : orthogonal_foci e) (h_area : triangle_area e) : e.b = 3 := by
  sorry


end ellipse_b_value_l738_73861


namespace arithmetic_computation_l738_73848

theorem arithmetic_computation : (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / -2) = -77 := by
  sorry

end arithmetic_computation_l738_73848


namespace permutation_residue_systems_l738_73812

theorem permutation_residue_systems (n : ℕ) : 
  (∃ p : Fin n → Fin n, Function.Bijective p ∧ 
    (∀ (i : Fin n), ∃ (j : Fin n), (p j + j : ℕ) % n = i) ∧
    (∀ (i : Fin n), ∃ (j : Fin n), (p j - j : ℤ) % n = i)) ↔ 
  (n % 6 = 1 ∨ n % 6 = 5) :=
sorry

end permutation_residue_systems_l738_73812


namespace piggy_bank_pennies_l738_73863

/-- Calculates the total number of pennies in a piggy bank after adding extra pennies -/
theorem piggy_bank_pennies (compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) :
  compartments = 20 →
  initial_pennies = 10 →
  added_pennies = 15 →
  compartments * (initial_pennies + added_pennies) = 500 := by
sorry

end piggy_bank_pennies_l738_73863


namespace min_value_theorem_l738_73832

theorem min_value_theorem (x y : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : x + y = 6) :
  ((x - 1)^2 / (y - 2)) + ((y - 1)^2 / (x - 2)) ≥ 8 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ x₀ + y₀ = 6 ∧
    ((x₀ - 1)^2 / (y₀ - 2)) + ((y₀ - 1)^2 / (x₀ - 2)) = 8 :=
by sorry

end min_value_theorem_l738_73832


namespace bike_ride_distance_l738_73827

theorem bike_ride_distance (first_hour second_hour third_hour : ℝ) : 
  second_hour = first_hour * 1.2 →
  third_hour = second_hour * 1.25 →
  first_hour + second_hour + third_hour = 74 →
  second_hour = 24 := by
sorry

end bike_ride_distance_l738_73827


namespace sequence_property_l738_73889

theorem sequence_property (a : ℕ → ℝ) :
  a 2 = 2 ∧ (∀ n : ℕ, n ≥ 2 → a (n + 1) - a n - 1 = 0) →
  ∀ n : ℕ, n ≥ 2 → a n = n :=
by sorry

end sequence_property_l738_73889


namespace bird_feeder_problem_l738_73814

theorem bird_feeder_problem (feeder_capacity : ℝ) (birds_per_cup : ℝ) (stolen_amount : ℝ) :
  feeder_capacity = 2 ∧ birds_per_cup = 14 ∧ stolen_amount = 0.5 →
  (feeder_capacity - stolen_amount) * birds_per_cup = 21 := by
  sorry

end bird_feeder_problem_l738_73814


namespace spending_problem_l738_73844

theorem spending_problem (initial_amount : ℚ) : 
  (2 / 5 : ℚ) * initial_amount = 600 → initial_amount = 1500 := by
  sorry

end spending_problem_l738_73844


namespace lesser_fraction_l738_73845

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 3/4) (h_product : x * y = 1/8) : 
  min x y = 1/4 := by sorry

end lesser_fraction_l738_73845


namespace swing_wait_time_l738_73802

/-- Proves that the wait time for swings is 4.75 minutes given the problem conditions -/
theorem swing_wait_time :
  let kids_for_swings : ℕ := 3
  let kids_for_slide : ℕ := 6
  let slide_wait_time : ℝ := 15
  let wait_time_difference : ℝ := 270
  let swing_wait_time : ℝ := (slide_wait_time + wait_time_difference) / 60
  swing_wait_time = 4.75 := by
sorry

#eval (15 + 270) / 60  -- Should output 4.75

end swing_wait_time_l738_73802


namespace revenue_change_l738_73894

theorem revenue_change 
  (price_increase : ℝ) 
  (sales_decrease : ℝ) 
  (price_increase_percent : price_increase = 30) 
  (sales_decrease_percent : sales_decrease = 20) : 
  (1 + price_increase / 100) * (1 - sales_decrease / 100) - 1 = 0.04 := by
sorry

end revenue_change_l738_73894


namespace intersection_of_lines_l738_73870

theorem intersection_of_lines :
  ∃! (x y : ℚ), (6 * x - 5 * y = 10) ∧ (8 * x + 2 * y = 20) ∧ (x = 30 / 13) ∧ (y = 10 / 13) := by
  sorry

end intersection_of_lines_l738_73870


namespace winning_candidate_percentage_l738_73859

def candidate1_votes : ℕ := 1136
def candidate2_votes : ℕ := 8236
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

theorem winning_candidate_percentage :
  (winning_votes : ℚ) / (total_votes : ℚ) * 100 = 58.14 := by
  sorry

end winning_candidate_percentage_l738_73859


namespace largest_number_with_quotient_30_l738_73807

theorem largest_number_with_quotient_30 : 
  ∀ n : ℕ, n ≤ 960 ∧ n / 31 = 30 → n = 960 :=
by
  sorry

end largest_number_with_quotient_30_l738_73807


namespace hyperbola_real_axis_length_l738_73887

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 - y^2 = a^2

-- Define the semi-latus rectum of a parabola
def semi_latus_rectum (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

theorem hyperbola_real_axis_length 
  (a p x1 y1 x2 y2 : ℝ) 
  (h1 : hyperbola a x1 y1)
  (h2 : hyperbola a x2 y2)
  (h3 : semi_latus_rectum p x1)
  (h4 : semi_latus_rectum p x2)
  (h5 : distance x1 y1 x2 y2 = 4 * (3^(1/2))) :
  2 * a = 4 := by
sorry

end hyperbola_real_axis_length_l738_73887


namespace min_value_problem_l738_73817

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f a b c x ≥ 4) 
  (hmin_exists : ∃ x, f a b c x = 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end min_value_problem_l738_73817


namespace andre_purchase_total_l738_73841

/-- Calculates the discounted price given the original price and discount percentage. -/
def discountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Calculates the total price for multiple items with the same price. -/
def totalPrice (itemPrice : ℚ) (quantity : ℕ) : ℚ :=
  itemPrice * quantity

theorem andre_purchase_total : 
  let treadmillOriginalPrice : ℚ := 1350
  let treadmillDiscount : ℚ := 30
  let plateOriginalPrice : ℚ := 60
  let plateQuantity : ℕ := 2
  let plateDiscount : ℚ := 15
  
  discountedPrice treadmillOriginalPrice treadmillDiscount + 
  discountedPrice (totalPrice plateOriginalPrice plateQuantity) plateDiscount = 1047 := by
sorry

end andre_purchase_total_l738_73841


namespace abs_sum_gt_abs_diff_when_product_positive_l738_73804

theorem abs_sum_gt_abs_diff_when_product_positive (a b : ℝ) (h : a * b > 0) :
  |a + b| > |a - b| := by sorry

end abs_sum_gt_abs_diff_when_product_positive_l738_73804


namespace maple_pine_height_difference_l738_73878

/-- The height of the pine tree in feet -/
def pine_height : ℚ := 24 + 1/4

/-- The height of the maple tree in feet -/
def maple_height : ℚ := 31 + 2/3

/-- The difference in height between the maple and pine trees -/
def height_difference : ℚ := maple_height - pine_height

theorem maple_pine_height_difference :
  height_difference = 7 + 5/12 := by sorry

end maple_pine_height_difference_l738_73878


namespace consecutive_integers_base_sum_l738_73813

/-- Represents a number in a given base -/
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem consecutive_integers_base_sum (C D : Nat) : 
  C.succ = D →
  C < D →
  to_base_10 [2, 3, 1] C + to_base_10 [5, 6] D = to_base_10 [1, 0, 5] (C + D) →
  C + D = 7 := by sorry

end consecutive_integers_base_sum_l738_73813


namespace group_earnings_l738_73829

/-- Represents the wage of a man in rupees -/
def man_wage : ℕ := 6

/-- Represents the number of men in the group -/
def num_men : ℕ := 5

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 8

/-- Represents the number of women in the group (unknown) -/
def num_women : ℕ := sorry

/-- The total amount earned by the group -/
def total_amount : ℕ := 3 * (num_men * man_wage)

theorem group_earnings : 
  total_amount = 90 := by sorry

end group_earnings_l738_73829


namespace diamond_digit_value_l738_73809

/-- Given that ◇6₉ = ◇3₁₀, where ◇ represents a digit, prove that ◇ = 3 -/
theorem diamond_digit_value :
  ∀ (diamond : ℕ),
  diamond < 10 →
  diamond * 9 + 6 = diamond * 10 + 3 →
  diamond = 3 :=
by
  sorry

end diamond_digit_value_l738_73809


namespace grade_distribution_l738_73865

theorem grade_distribution (n : ℕ) : 
  ∃ (a b c m : ℕ),
    (2 * m + 3 = n) ∧  -- Total students
    (b = a + 2) ∧      -- B grades
    (c = 2 * b) ∧      -- C grades
    (4 * a + 6 ≠ n)    -- Total A, B, C grades ≠ Total students
  := by sorry

end grade_distribution_l738_73865


namespace arithmetic_sequence_problem_l738_73881

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + a_4 + a_10 + a_16 + a_19 = 150,
    prove that a_18 - 2a_14 = -30 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 1 + a 4 + a 10 + a 16 + a 19 = 150) : 
    a 18 - 2 * a 14 = -30 := by
  sorry

end arithmetic_sequence_problem_l738_73881


namespace raghu_investment_l738_73896

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  raghu + trishul + vishal = 6069 →
  raghu = 2100 := by
sorry

end raghu_investment_l738_73896


namespace triangle_angle_problem_l738_73800

theorem triangle_angle_problem (A B C x : ℝ) : 
  A = 40 ∧ B = 3*x ∧ C = 2*x ∧ A + B + C = 180 → x = 28 := by
  sorry

end triangle_angle_problem_l738_73800


namespace min_value_theorem_l738_73868

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 8/n = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 8/y = 4 → 8*m + n ≤ 8*x + y :=
by sorry

end min_value_theorem_l738_73868


namespace final_ratio_theorem_l738_73898

/-- Represents the final amount ratio between two players -/
structure FinalRatio where
  player1 : ℕ
  player2 : ℕ

/-- Represents a game with three players -/
structure Game where
  initialAmount : ℕ
  finalRatioAS : FinalRatio
  sGain : ℕ

theorem final_ratio_theorem (g : Game) 
  (h1 : g.initialAmount = 70)
  (h2 : g.finalRatioAS = FinalRatio.mk 1 2)
  (h3 : g.sGain = 50) :
  ∃ (finalRatioSB : FinalRatio), 
    finalRatioSB.player1 = 4 ∧ 
    finalRatioSB.player2 = 1 := by
  sorry

end final_ratio_theorem_l738_73898


namespace scientific_notation_of_seven_nm_l738_73880

-- Define the value of 7nm in meters
def seven_nm : ℝ := 0.000000007

-- Theorem to prove the scientific notation
theorem scientific_notation_of_seven_nm :
  ∃ (a : ℝ) (n : ℤ), seven_nm = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
by sorry

end scientific_notation_of_seven_nm_l738_73880


namespace unique_pair_existence_l738_73833

theorem unique_pair_existence : ∃! (c d : Real),
  c ∈ Set.Ioo 0 (Real.pi / 2) ∧
  d ∈ Set.Ioo 0 (Real.pi / 2) ∧
  c < d ∧
  Real.sin (Real.cos c) = c ∧
  Real.cos (Real.sin d) = d := by
  sorry

end unique_pair_existence_l738_73833


namespace prime_square_mod_twelve_l738_73869

theorem prime_square_mod_twelve (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end prime_square_mod_twelve_l738_73869


namespace prob_at_least_9_is_0_7_l738_73811

/-- A shooter has probabilities of scoring different points in one shot. -/
structure Shooter where
  prob_10 : ℝ  -- Probability of scoring 10 points
  prob_9 : ℝ   -- Probability of scoring 9 points
  prob_8_or_less : ℝ  -- Probability of scoring 8 or fewer points
  sum_to_one : prob_10 + prob_9 + prob_8_or_less = 1  -- Probabilities sum to 1

/-- The probability of scoring at least 9 points is the sum of probabilities of scoring 10 and 9 points. -/
def prob_at_least_9 (s : Shooter) : ℝ := s.prob_10 + s.prob_9

/-- Given the probabilities for a shooter, prove that the probability of scoring at least 9 points is 0.7. -/
theorem prob_at_least_9_is_0_7 (s : Shooter) 
    (h1 : s.prob_10 = 0.4) 
    (h2 : s.prob_9 = 0.3) 
    (h3 : s.prob_8_or_less = 0.3) : 
  prob_at_least_9 s = 0.7 := by
  sorry

end prob_at_least_9_is_0_7_l738_73811


namespace quadratic_function_properties_l738_73803

-- Define the quadratic function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties (b c : ℝ) :
  (∀ α : ℝ, f b c (Real.sin α) ≥ 0) →
  (∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0) →
  (∃ M : ℝ, M = 8 ∧ ∀ α : ℝ, f b c (Real.sin α) ≤ M) →
  b = -4 ∧ c = 3 := by
  sorry

end quadratic_function_properties_l738_73803


namespace second_smallest_odd_number_l738_73857

/-- Given a sequence of four consecutive odd numbers whose sum is 112,
    the second smallest number in this sequence is 27. -/
theorem second_smallest_odd_number : ∀ (a b c d : ℤ),
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7) →  -- consecutive odd numbers
  (a + b + c + d = 112) →                                            -- sum is 112
  b = 27                                                             -- second smallest is 27
:= by sorry

end second_smallest_odd_number_l738_73857


namespace cost_per_crayon_is_two_l738_73831

/-- The number of crayons in half a dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens Jamal bought -/
def number_of_half_dozens : ℕ := 4

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := 48

/-- The total number of crayons Jamal bought -/
def total_crayons : ℕ := number_of_half_dozens * half_dozen

/-- The cost per crayon in dollars -/
def cost_per_crayon : ℚ := total_cost / total_crayons

theorem cost_per_crayon_is_two : cost_per_crayon = 2 := by
  sorry

end cost_per_crayon_is_two_l738_73831


namespace lcm_problem_l738_73840

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by sorry

end lcm_problem_l738_73840


namespace inequality_implication_l738_73853

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_implication_l738_73853


namespace intersection_y_coordinate_l738_73842

/-- The y-coordinate of the intersection point between a line and a parabola -/
theorem intersection_y_coordinate (x : ℝ) : 
  x > 0 ∧ 
  (x - 1)^2 + 1 = -2*x + 11 → 
  -2*x + 11 = 5 := by
  sorry

end intersection_y_coordinate_l738_73842


namespace original_fraction_problem_l738_73826

theorem original_fraction_problem (N D : ℚ) :
  (N > 0) →
  (D > 0) →
  ((1.4 * N) / (0.5 * D) = 4/5) →
  (N / D = 2/7) :=
by sorry

end original_fraction_problem_l738_73826


namespace book_sale_profit_l738_73858

/-- Prove that given two books with a total cost of 480, where the first book costs 280 and is sold
    at a 15% loss, and both books are sold at the same price, the gain percentage on the second book
    is 19%. -/
theorem book_sale_profit (total_cost : ℝ) (cost_book1 : ℝ) (loss_percentage : ℝ) 
  (h1 : total_cost = 480)
  (h2 : cost_book1 = 280)
  (h3 : loss_percentage = 15)
  (h4 : ∃ (sell_price : ℝ), sell_price = cost_book1 * (1 - loss_percentage / 100) ∧ 
        sell_price = (total_cost - cost_book1) * (1 + x / 100)) :
  x = 19 := by
  sorry

end book_sale_profit_l738_73858


namespace total_balloons_count_l738_73818

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 60

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 85

/-- The number of blue balloons Alex has -/
def alex_balloons : ℕ := 37

/-- The total number of blue balloons -/
def total_balloons : ℕ := joan_balloons + melanie_balloons + alex_balloons

theorem total_balloons_count : total_balloons = 182 := by
  sorry

end total_balloons_count_l738_73818


namespace pure_imaginary_complex_product_l738_73866

theorem pure_imaginary_complex_product (a : ℝ) :
  let z : ℂ := (1 - 2*I) * (a - I) * I
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
sorry

end pure_imaginary_complex_product_l738_73866


namespace fraction_zero_solution_l738_73810

theorem fraction_zero_solution (x : ℝ) : 
  (x + 2) / (2 * x - 4) = 0 ↔ x = -2 ∧ 2 * x - 4 ≠ 0 :=
by sorry

end fraction_zero_solution_l738_73810
