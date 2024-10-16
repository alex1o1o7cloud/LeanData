import Mathlib

namespace NUMINAMATH_CALUDE_investment_percentage_l2109_210957

/-- Proves that given the investment conditions, the unknown percentage is 4% -/
theorem investment_percentage (total_investment : ℝ) (known_rate : ℝ) (unknown_rate : ℝ) 
  (total_interest : ℝ) (amount_at_unknown_rate : ℝ) :
  total_investment = 17000 →
  known_rate = 18 →
  total_interest = 1380 →
  amount_at_unknown_rate = 12000 →
  (amount_at_unknown_rate * unknown_rate / 100 + 
   (total_investment - amount_at_unknown_rate) * known_rate / 100 = total_interest) →
  unknown_rate = 4 := by
sorry


end NUMINAMATH_CALUDE_investment_percentage_l2109_210957


namespace NUMINAMATH_CALUDE_event_X_6_equivalent_to_draw_6_and_two_others_l2109_210988

/-- Represents a ball with a number -/
structure Ball :=
  (number : Nat)

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of balls to be drawn -/
def numDrawn : Nat := 3

/-- X represents the highest number on the drawn balls -/
def X (drawn : Finset Ball) : Nat := sorry

/-- The event where X equals 6 -/
def event_X_equals_6 (drawn : Finset Ball) : Prop :=
  X drawn = 6

/-- The event of drawing 3 balls with one numbered 6 and two others from 1 to 5 -/
def event_draw_6_and_two_others (drawn : Finset Ball) : Prop := sorry

theorem event_X_6_equivalent_to_draw_6_and_two_others :
  ∀ drawn : Finset Ball,
  drawn.card = numDrawn →
  (event_X_equals_6 drawn ↔ event_draw_6_and_two_others drawn) :=
sorry

end NUMINAMATH_CALUDE_event_X_6_equivalent_to_draw_6_and_two_others_l2109_210988


namespace NUMINAMATH_CALUDE_dot_product_equals_eight_l2109_210996

def a : Fin 2 → ℝ := ![0, 4]
def b : Fin 2 → ℝ := ![2, 2]

theorem dot_product_equals_eight :
  (Finset.univ.sum (λ i => a i * b i)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_eight_l2109_210996


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l2109_210920

def S (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4*n - 3) + if n > 1 then S (n-1) else 0

theorem sequence_sum_problem : S 15 + S 22 - S 31 = -76 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l2109_210920


namespace NUMINAMATH_CALUDE_no_valid_labeling_exists_l2109_210977

/-- Represents a simple undirected graph with 6 vertices -/
structure Graph :=
  (edges : Set (Fin 6 × Fin 6))
  (symmetric : ∀ (a b : Fin 6), (a, b) ∈ edges → (b, a) ∈ edges)
  (irreflexive : ∀ (a : Fin 6), (a, a) ∉ edges)

/-- A function assigning natural numbers to vertices -/
def VertexLabeling := Fin 6 → ℕ+

/-- Checks if the labeling satisfies the divisibility condition for the given graph -/
def ValidLabeling (g : Graph) (f : VertexLabeling) : Prop :=
  (∀ (a b : Fin 6), (a, b) ∈ g.edges → (f a ∣ f b) ∨ (f b ∣ f a)) ∧
  (∀ (a b : Fin 6), a ≠ b → (a, b) ∉ g.edges → ¬(f a ∣ f b) ∧ ¬(f b ∣ f a))

/-- The main theorem stating that no valid labeling exists for any graph with 6 vertices -/
theorem no_valid_labeling_exists : ∀ (g : Graph), ¬∃ (f : VertexLabeling), ValidLabeling g f := by
  sorry

end NUMINAMATH_CALUDE_no_valid_labeling_exists_l2109_210977


namespace NUMINAMATH_CALUDE_a_10_equals_21_l2109_210959

def arithmetic_sequence (b : ℕ+ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ+, b (n + 1) - b n = d

theorem a_10_equals_21
  (a : ℕ+ → ℚ)
  (b : ℕ+ → ℚ)
  (h1 : a 1 = 3)
  (h2 : arithmetic_sequence b)
  (h3 : ∀ n : ℕ+, b n = a (n + 1) - a n)
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 10 = 21 := by
sorry

end NUMINAMATH_CALUDE_a_10_equals_21_l2109_210959


namespace NUMINAMATH_CALUDE_sum_squared_equals_400_l2109_210995

variable (a b c : ℝ)

theorem sum_squared_equals_400 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a*b + b*c + c*a = 5) : 
  (a + b + c)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_equals_400_l2109_210995


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_determination_l2109_210953

theorem monic_cubic_polynomial_determination (q : ℝ → ℂ) :
  (∀ x, q x = x^3 + (q 1 - 3) * x^2 + (q 1 - 2 * (q 1 - 3)) * x + q 0) →
  q (2 - 3*I) = 0 →
  q 0 = -36 →
  ∀ x, q x = x^3 - (88/13) * x^2 + (325/13) * x - (468/13) :=
by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_determination_l2109_210953


namespace NUMINAMATH_CALUDE_fathers_age_l2109_210973

theorem fathers_age (man_age father_age : ℚ) : 
  man_age = (2 / 5) * father_age →
  man_age + 10 = (1 / 2) * (father_age + 10) →
  father_age = 50 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l2109_210973


namespace NUMINAMATH_CALUDE_midpoint_implies_equation_ratio_implies_equation_l2109_210963

/-- A line passing through point M(-2,1) and intersecting x and y axes at A and B respectively -/
structure Line :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (h_A : A.2 = 0)
  (h_B : B.1 = 0)

/-- The point M(-2,1) -/
def M : ℝ × ℝ := (-2, 1)

/-- M is the midpoint of AB -/
def is_midpoint (l : Line) : Prop :=
  M = ((l.A.1 + l.B.1) / 2, (l.A.2 + l.B.2) / 2)

/-- M divides AB in the ratio of 2:1 or 1:2 -/
def divides_in_ratio (l : Line) : Prop :=
  (M.1 - l.A.1, M.2 - l.A.2) = (2 * (l.B.1 - M.1), 2 * (l.B.2 - M.2)) ∨
  (M.1 - l.A.1, M.2 - l.A.2) = (-2 * (l.B.1 - M.1), -2 * (l.B.2 - M.2))

/-- The equation of the line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a b c : ℝ)

theorem midpoint_implies_equation (l : Line) (h : is_midpoint l) :
  ∃ (eq : LineEquation), eq.a * l.A.1 + eq.b * l.A.2 + eq.c = 0 ∧
                         eq.a * l.B.1 + eq.b * l.B.2 + eq.c = 0 ∧
                         eq.a * M.1 + eq.b * M.2 + eq.c = 0 ∧
                         eq.a = 1 ∧ eq.b = -2 ∧ eq.c = 4 := by sorry

theorem ratio_implies_equation (l : Line) (h : divides_in_ratio l) :
  ∃ (eq1 eq2 : LineEquation),
    (eq1.a * l.A.1 + eq1.b * l.A.2 + eq1.c = 0 ∧
     eq1.a * l.B.1 + eq1.b * l.B.2 + eq1.c = 0 ∧
     eq1.a * M.1 + eq1.b * M.2 + eq1.c = 0 ∧
     eq1.a = 1 ∧ eq1.b = -4 ∧ eq1.c = 6) ∨
    (eq2.a * l.A.1 + eq2.b * l.A.2 + eq2.c = 0 ∧
     eq2.a * l.B.1 + eq2.b * l.B.2 + eq2.c = 0 ∧
     eq2.a * M.1 + eq2.b * M.2 + eq2.c = 0 ∧
     eq2.a = 1 ∧ eq2.b = 4 ∧ eq2.c = -2) := by sorry

end NUMINAMATH_CALUDE_midpoint_implies_equation_ratio_implies_equation_l2109_210963


namespace NUMINAMATH_CALUDE_g_symmetric_about_one_l2109_210907

-- Define the real-valued functions f and g
variable (f : ℝ → ℝ)
def g (x : ℝ) : ℝ := f (|x - 1|)

-- Define symmetry about a vertical line
def symmetric_about_line (h : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, h (a + x) = h (a - x)

-- Theorem statement
theorem g_symmetric_about_one (f : ℝ → ℝ) :
  symmetric_about_line (g f) 1 := by
  sorry

end NUMINAMATH_CALUDE_g_symmetric_about_one_l2109_210907


namespace NUMINAMATH_CALUDE_track_length_proof_l2109_210925

/-- The length of the circular track in meters -/
def track_length : ℝ := 180

/-- The distance Brenda runs before their first meeting in meters -/
def brenda_first_meeting : ℝ := 120

/-- The additional distance Sally runs after their first meeting before their second meeting in meters -/
def sally_additional : ℝ := 180

theorem track_length_proof :
  ∃ (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 ∧ sally_speed > 0 ∧
    brenda_speed ≠ sally_speed ∧
    (sally_speed * track_length = brenda_speed * (track_length + brenda_first_meeting)) ∧
    (sally_speed * (track_length + sally_additional) = brenda_speed * (2 * track_length + brenda_first_meeting)) :=
sorry

end NUMINAMATH_CALUDE_track_length_proof_l2109_210925


namespace NUMINAMATH_CALUDE_prob_at_least_two_tails_is_half_l2109_210984

/-- The probability of getting at least 2 tails when tossing 3 fair coins -/
def prob_at_least_two_tails : ℚ := 1/2

/-- The number of possible outcomes when tossing 3 coins -/
def total_outcomes : ℕ := 2^3

/-- The number of favorable outcomes (at least 2 tails) -/
def favorable_outcomes : ℕ := 4

theorem prob_at_least_two_tails_is_half :
  prob_at_least_two_tails = (favorable_outcomes : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_tails_is_half_l2109_210984


namespace NUMINAMATH_CALUDE_susie_house_rooms_l2109_210923

/-- The number of rooms in Susie's house -/
def number_of_rooms : ℕ := 6

/-- The time it takes Susie to vacuum the whole house (in hours) -/
def total_vacuum_time : ℚ := 2

/-- The time it takes Susie to vacuum one room (in minutes) -/
def time_per_room : ℕ := 20

/-- Theorem stating that the number of rooms in Susie's house is 6 -/
theorem susie_house_rooms :
  number_of_rooms = (total_vacuum_time * 60) / time_per_room := by
  sorry

end NUMINAMATH_CALUDE_susie_house_rooms_l2109_210923


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l2109_210946

theorem max_value_x_minus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2*y) :
  ∃ (max : ℝ), max = 2/3 ∧ x - 2*y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l2109_210946


namespace NUMINAMATH_CALUDE_wolf_and_nobel_count_l2109_210974

/-- Represents the number of scientists in various categories at a workshop --/
structure WorkshopAttendees where
  total : ℕ
  wolf : ℕ
  nobel : ℕ
  wolf_and_nobel : ℕ

/-- The conditions of the workshop --/
def workshop : WorkshopAttendees where
  total := 50
  wolf := 31
  nobel := 25
  wolf_and_nobel := 0  -- This is what we need to prove

/-- Theorem stating the number of scientists who were both Wolf and Nobel laureates --/
theorem wolf_and_nobel_count (w : WorkshopAttendees) 
  (h1 : w.total = 50)
  (h2 : w.wolf = 31)
  (h3 : w.nobel = 25)
  (h4 : w.nobel - w.wolf = 3 + (w.total - w.nobel - (w.wolf - w.wolf_and_nobel))) :
  w.wolf_and_nobel = 3 := by
  sorry

end NUMINAMATH_CALUDE_wolf_and_nobel_count_l2109_210974


namespace NUMINAMATH_CALUDE_equation_solution_l2109_210917

theorem equation_solution (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 * k = 24 ↔ 5 * x + 3 = 0) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2109_210917


namespace NUMINAMATH_CALUDE_square_difference_division_eleven_l2109_210919

theorem square_difference_division_eleven : (121^2 - 110^2) / 11 = 231 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_eleven_l2109_210919


namespace NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_for_abs_m_equals_one_l2109_210942

theorem m_equals_one_sufficient_not_necessary_for_abs_m_equals_one :
  (∀ m : ℝ, m = 1 → |m| = 1) ∧
  (∃ m : ℝ, |m| = 1 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_for_abs_m_equals_one_l2109_210942


namespace NUMINAMATH_CALUDE_chef_leftover_potatoes_l2109_210982

theorem chef_leftover_potatoes 
  (fries_per_potato : ℕ) 
  (total_potatoes : ℕ) 
  (required_fries : ℕ) 
  (h1 : fries_per_potato = 25)
  (h2 : total_potatoes = 15)
  (h3 : required_fries = 200) :
  total_potatoes - (required_fries / fries_per_potato) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_chef_leftover_potatoes_l2109_210982


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2109_210945

theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = (3/5) * x ∧ x^2 / a^2 - y^2 / 9 = 1) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2109_210945


namespace NUMINAMATH_CALUDE_angle_330_equals_negative_30_l2109_210971

/-- Two angles have the same terminal side if they differ by a multiple of 360° --/
def same_terminal_side (α β : Real) : Prop :=
  ∃ k : Int, α = β + 360 * k

/-- The problem statement --/
theorem angle_330_equals_negative_30 :
  same_terminal_side 330 (-30) := by
  sorry

end NUMINAMATH_CALUDE_angle_330_equals_negative_30_l2109_210971


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l2109_210912

theorem factorization_of_2x_squared_minus_18 (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l2109_210912


namespace NUMINAMATH_CALUDE_t_value_l2109_210955

theorem t_value : 
  let t := 2 / (1 - Real.rpow 2 (1/3))
  t = -2 * (1 + Real.rpow 2 (1/3) + Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_t_value_l2109_210955


namespace NUMINAMATH_CALUDE_cousin_reading_time_l2109_210980

/-- Given reading speeds and book lengths, calculate cousin's reading time -/
theorem cousin_reading_time
  (my_speed : ℝ)
  (my_time : ℝ)
  (my_book_length : ℝ)
  (cousin_speed_ratio : ℝ)
  (cousin_book_length_ratio : ℝ)
  (h1 : my_time = 180) -- 3 hours in minutes
  (h2 : cousin_speed_ratio = 5)
  (h3 : cousin_book_length_ratio = 1.5)
  : (cousin_book_length_ratio * my_book_length) / (cousin_speed_ratio * my_speed) = 54 := by
  sorry

end NUMINAMATH_CALUDE_cousin_reading_time_l2109_210980


namespace NUMINAMATH_CALUDE_rectangle_area_with_three_squares_l2109_210947

/-- Given three non-overlapping squares where one has an area of 4 square inches,
    and another has double the side length of the first two,
    the total area of the rectangle containing all three squares is 24 square inches. -/
theorem rectangle_area_with_three_squares (s₁ s₂ s₃ : Real) : 
  s₁ * s₁ = 4 →  -- Area of the first square (shaded) is 4
  s₂ = s₁ →      -- Second square has the same side length as the first
  s₃ = 2 * s₁ →  -- Third square has double the side length of the first two
  s₁ * s₁ + s₂ * s₂ + s₃ * s₃ = 24 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_with_three_squares_l2109_210947


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2109_210979

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 12 →
  (1/2) * a * b = 6 →
  a^2 + b^2 = c^2 →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2109_210979


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2109_210970

theorem quadratic_inequality_solution (t a : ℝ) : 
  (∀ x, tx^2 - 6*x + t^2 < 0 ↔ x ∈ Set.Ioi 1 ∪ Set.Iic a) →
  (t*a^2 - 6*a + t^2 = 0 ∧ t*1^2 - 6*1 + t^2 = 0) →
  t < 0 →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2109_210970


namespace NUMINAMATH_CALUDE_digit_equation_solution_l2109_210999

theorem digit_equation_solution :
  ∀ (A M C : ℕ),
    A ≤ 9 → M ≤ 9 → C ≤ 9 →
    (100 * A + 10 * M + C) * (2 * (A + M + C + 1)) = 4010 →
    A = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l2109_210999


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2109_210937

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

-- State the theorem
theorem quadratic_symmetry (a : ℝ) :
  (f a 1 = 2) → (f a (-1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2109_210937


namespace NUMINAMATH_CALUDE_austen_gathering_handshakes_l2109_210927

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  men_shake_all_but_spouse : Bool
  women_shake_count : ℕ

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  let total_people := 2 * g.couples
  let men_count := g.couples
  let women_count := g.couples
  let men_handshakes := men_count.choose 2
  let men_women_handshakes := if g.men_shake_all_but_spouse then men_count * (women_count - 1) else 0
  men_handshakes + men_women_handshakes + g.women_shake_count

theorem austen_gathering_handshakes :
  let g : Gathering := { couples := 15, men_shake_all_but_spouse := true, women_shake_count := 1 }
  total_handshakes g = 316 := by
  sorry

end NUMINAMATH_CALUDE_austen_gathering_handshakes_l2109_210927


namespace NUMINAMATH_CALUDE_prob_fully_black_after_two_rotations_l2109_210940

/-- Represents a 3x3 grid where each cell can be either black or white -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Represents a 90-degree clockwise rotation of the grid -/
def rotate (g : Grid) : Grid := sorry

/-- Applies the painting rule after rotation -/
def paint_after_rotate (g : Grid) : Grid := sorry

/-- The probability of a single square being or becoming black after one rotation and painting -/
def prob_black_after_one : ℚ := 3/4

/-- The probability of the center square being initially black -/
def prob_center_black : ℚ := 1/2

/-- Theorem stating the probability of the grid being fully black after two rotations -/
theorem prob_fully_black_after_two_rotations :
  (prob_center_black * prob_black_after_one ^ 8 : ℚ) = 6561/131072 := by sorry

end NUMINAMATH_CALUDE_prob_fully_black_after_two_rotations_l2109_210940


namespace NUMINAMATH_CALUDE_sunflower_majority_day_two_l2109_210916

/-- Represents the proportion of sunflower seeds in the feeder on a given day -/
def sunflower_proportion (day : ℕ) : ℝ :=
  1 - (0.6 : ℝ) ^ day

/-- The daily seed mixture contains 40% sunflower seeds -/
axiom seed_mixture : (0.4 : ℝ) = 1 - (0.6 : ℝ)

/-- Birds eat 40% of sunflower seeds daily -/
axiom bird_consumption : (0.6 : ℝ) = 1 - (0.4 : ℝ)

/-- Theorem: On day 2, more than half the seeds are sunflower seeds -/
theorem sunflower_majority_day_two :
  sunflower_proportion 2 > (0.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sunflower_majority_day_two_l2109_210916


namespace NUMINAMATH_CALUDE_train_passing_time_l2109_210936

theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 350)
  (h2 : length2 = 450)
  (h3 : speed1 = 63 * 1000 / 3600)
  (h4 : speed2 = 81 * 1000 / 3600)
  (h5 : speed2 > speed1) :
  (length1 + length2) / (speed2 - speed1) = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2109_210936


namespace NUMINAMATH_CALUDE_incorrect_statement_l2109_210931

theorem incorrect_statement : ¬(0 > |(-1)|) ∧ (-(-3) = 3) ∧ (|2| = |-2|) ∧ (-2 > -3) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2109_210931


namespace NUMINAMATH_CALUDE_states_fraction_1800_1809_l2109_210928

theorem states_fraction_1800_1809 (total_states : Nat) (states_1800_1809 : Nat) :
  total_states = 30 →
  states_1800_1809 = 5 →
  (states_1800_1809 : ℚ) / total_states = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_1800_1809_l2109_210928


namespace NUMINAMATH_CALUDE_points_on_line_l2109_210924

/-- Given three points (8, 10), (0, m), and (-8, 6) on a straight line, prove that m = 8 -/
theorem points_on_line (m : ℝ) : 
  (∀ (t : ℝ), ∃ (s : ℝ), (8 * (1 - t) + 0 * t, 10 * (1 - t) + m * t) = 
    (0 * (1 - s) + (-8) * s, m * (1 - s) + 6 * s)) → 
  m = 8 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l2109_210924


namespace NUMINAMATH_CALUDE_qrs_profit_change_l2109_210997

/-- Represents the profit change for QRS company over a quarter -/
structure ProfitChange where
  march_to_april : Real
  april_to_may : Real
  may_to_june : Real

/-- Calculates the total profit change from March to June -/
def total_change (pc : ProfitChange) : Real :=
  (1 + pc.march_to_april) * (1 + pc.april_to_may) * (1 + pc.may_to_june) - 1

/-- Theorem stating that given the specific profit changes, the total change is 80% -/
theorem qrs_profit_change :
  let pc : ProfitChange := {
    march_to_april := 0.5,
    april_to_may := -0.2,
    may_to_june := 0.5
  }
  total_change pc = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_qrs_profit_change_l2109_210997


namespace NUMINAMATH_CALUDE_bertha_family_childless_l2109_210952

/-- Represents the family structure of Bertha and her descendants -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_children : ℕ

/-- The properties of Bertha's family -/
def bertha_family_properties (f : BerthaFamily) : Prop :=
  f.daughters = 6 ∧
  f.granddaughters = 6 * f.daughters_with_children ∧
  f.daughters + f.granddaughters = 30

/-- The theorem stating the number of Bertha's daughters and granddaughters without children -/
theorem bertha_family_childless (f : BerthaFamily) 
  (h : bertha_family_properties f) : 
  f.daughters + f.granddaughters - f.daughters_with_children = 26 := by
  sorry


end NUMINAMATH_CALUDE_bertha_family_childless_l2109_210952


namespace NUMINAMATH_CALUDE_expression_simplification_l2109_210969

theorem expression_simplification (m : ℝ) (h : m = 5) :
  (3*m + 6) / (m^2 + 4*m + 4) / ((m - 2) / (m + 2)) + 1 / (2 - m) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2109_210969


namespace NUMINAMATH_CALUDE_system_one_solution_l2109_210993

theorem system_one_solution (x y : ℝ) : 
  x + 3 * y = 3 ∧ x - y = 1 → x = (3 : ℝ) / 2 ∧ y = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_system_one_solution_l2109_210993


namespace NUMINAMATH_CALUDE_notebook_distribution_l2109_210987

/-- Proves that the ratio of notebooks per child to the number of children is 1:8 
    given the conditions in the problem. -/
theorem notebook_distribution (C : ℕ) (N : ℚ) : 
  (∃ (k : ℕ), N = k / C) →  -- Number of notebooks each child got is a fraction of number of children
  (16 = 2 * k / C) →        -- If number of children halved, each would get 16 notebooks
  (C * N = 512) →           -- Total notebooks distributed is 512
  N / C = 1 / 8 :=          -- Ratio of notebooks per child to number of children is 1:8
by sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2109_210987


namespace NUMINAMATH_CALUDE_average_weight_abc_l2109_210901

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 48 →
  (b + c) / 2 = 42 →
  b = 51 →
  (a + b + c) / 3 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2109_210901


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2109_210968

theorem consecutive_integers_product_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 358800 → 
  n + (n + 1) + (n + 2) + (n + 3) = 98 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2109_210968


namespace NUMINAMATH_CALUDE_unique_solution_l2109_210967

theorem unique_solution (a b c d e : ℝ) :
  a ∈ Set.Icc (-2 : ℝ) 2 ∧
  b ∈ Set.Icc (-2 : ℝ) 2 ∧
  c ∈ Set.Icc (-2 : ℝ) 2 ∧
  d ∈ Set.Icc (-2 : ℝ) 2 ∧
  e ∈ Set.Icc (-2 : ℝ) 2 ∧
  a + b + c + d + e = 0 ∧
  a^3 + b^3 + c^3 + d^3 + e^3 = 0 ∧
  a^5 + b^5 + c^5 + d^5 + e^5 = 10 →
  ({a, b, c, d, e} : Set ℝ) = {2, (Real.sqrt 5 - 1) / 2, (Real.sqrt 5 - 1) / 2, -(Real.sqrt 5 + 1) / 2, -(Real.sqrt 5 + 1) / 2} :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2109_210967


namespace NUMINAMATH_CALUDE_mongolia_1980_imo_host_l2109_210918

/-- Represents countries in East Asia -/
inductive EastAsianCountry
  | China
  | Japan
  | Mongolia
  | NorthKorea
  | SouthKorea
  | Taiwan

/-- Represents the International Mathematical Olympiad event -/
structure IMOEvent where
  year : Nat
  host : EastAsianCountry
  canceled : Bool

/-- The 1980 IMO event -/
def imo1980 : IMOEvent :=
  { year := 1980
  , host := EastAsianCountry.Mongolia
  , canceled := true }

/-- Theorem stating that Mongolia was the scheduled host of the canceled 1980 IMO -/
theorem mongolia_1980_imo_host :
  imo1980.year = 1980 ∧
  imo1980.host = EastAsianCountry.Mongolia ∧
  imo1980.canceled = true :=
by sorry

end NUMINAMATH_CALUDE_mongolia_1980_imo_host_l2109_210918


namespace NUMINAMATH_CALUDE_A_characterization_and_inequality_l2109_210966

def f (x : ℝ) : ℝ := |2*x + 1| + |x - 2|

def A : Set ℝ := {x | f x < 3}

theorem A_characterization_and_inequality :
  (A = {x : ℝ | -2/3 < x ∧ x < 0}) ∧
  (∀ s t : ℝ, s ∈ A → t ∈ A → |1 - t/s| < |t - 1/s|) := by sorry

end NUMINAMATH_CALUDE_A_characterization_and_inequality_l2109_210966


namespace NUMINAMATH_CALUDE_craigs_walk_distance_l2109_210930

theorem craigs_walk_distance (distance_school_to_david : ℝ) (distance_david_to_home : ℝ)
  (h1 : distance_school_to_david = 0.27)
  (h2 : distance_david_to_home = 0.73) :
  distance_school_to_david + distance_david_to_home = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_craigs_walk_distance_l2109_210930


namespace NUMINAMATH_CALUDE_total_marks_math_physics_l2109_210954

/-- Given a student's marks in mathematics, physics, and chemistry, prove that
    the total marks in mathematics and physics is 60, under the given conditions. -/
theorem total_marks_math_physics (M P C : ℕ) : 
  C = P + 10 →  -- Chemistry score is 10 more than Physics
  (M + C) / 2 = 35 →  -- Average of Mathematics and Chemistry is 35
  M + P = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_math_physics_l2109_210954


namespace NUMINAMATH_CALUDE_derivative_bound_l2109_210992

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem derivative_bound
  (h_cont : ContDiff ℝ 3 f)
  (h_pos : ∀ x, f x > 0 ∧ (deriv f) x > 0 ∧ (deriv^[2] f) x > 0 ∧ (deriv^[3] f) x > 0)
  (h_bound : ∀ x, (deriv^[3] f) x ≤ f x) :
  ∀ x, (deriv f) x < 2 * f x :=
sorry

end NUMINAMATH_CALUDE_derivative_bound_l2109_210992


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2109_210978

theorem intersection_of_lines :
  ∃! p : ℝ × ℝ, 
    5 * p.1 - 3 * p.2 = 15 ∧ 
    6 * p.1 + 2 * p.2 = 14 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2109_210978


namespace NUMINAMATH_CALUDE_soccer_balls_count_initial_balls_count_l2109_210958

/-- The initial number of soccer balls in the bag -/
def initial_balls : ℕ := sorry

/-- The number of additional balls added to the bag -/
def added_balls : ℕ := 18

/-- The final number of balls in the bag -/
def final_balls : ℕ := 24

/-- Theorem stating that the initial number of balls plus the added balls equals the final number of balls -/
theorem soccer_balls_count : initial_balls + added_balls = final_balls := by sorry

/-- Theorem proving that the initial number of balls is 6 -/
theorem initial_balls_count : initial_balls = 6 := by sorry

end NUMINAMATH_CALUDE_soccer_balls_count_initial_balls_count_l2109_210958


namespace NUMINAMATH_CALUDE_max_value_implies_m_equals_20_l2109_210932

/-- The function f(x) = -x^3 + 6x^2 - m --/
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 - m

/-- The maximum value of f(x) is 12 --/
def max_value : ℝ := 12

theorem max_value_implies_m_equals_20 :
  (∃ x₀ : ℝ, ∀ x : ℝ, f x m ≤ f x₀ m) ∧ (∃ x₁ : ℝ, f x₁ m = max_value) → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_equals_20_l2109_210932


namespace NUMINAMATH_CALUDE_jane_sunflower_seeds_l2109_210964

/-- Given that Jane has 9 cans and places 6 seeds in each can,
    prove that the total number of sunflower seeds is 54. -/
theorem jane_sunflower_seeds (num_cans : ℕ) (seeds_per_can : ℕ) 
    (h1 : num_cans = 9) 
    (h2 : seeds_per_can = 6) : 
  num_cans * seeds_per_can = 54 := by
  sorry

end NUMINAMATH_CALUDE_jane_sunflower_seeds_l2109_210964


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2109_210910

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistanceTraveled (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  let descentDistances := List.range (bounces + 1) |>.map (fun i => initialHeight * reboundRatio ^ i)
  let ascentDistances := descentDistances.tail
  (descentDistances.sum + ascentDistances.sum)

/-- The theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistanceTraveled 200 (1/3) 4 = 397 + 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2109_210910


namespace NUMINAMATH_CALUDE_seniors_in_sample_is_fifty_l2109_210949

/-- Represents a school population with stratified sampling -/
structure SchoolPopulation where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  (senior_le_total : senior_students ≤ total_students)
  (sample_le_total : sample_size ≤ total_students)

/-- Calculates the number of senior students in a stratified sample -/
def seniors_in_sample (school : SchoolPopulation) : ℕ :=
  (school.senior_students * school.sample_size) / school.total_students

/-- Theorem stating that for the given school population, the number of seniors in the sample is 50 -/
theorem seniors_in_sample_is_fifty (school : SchoolPopulation)
  (h1 : school.total_students = 2000)
  (h2 : school.senior_students = 500)
  (h3 : school.sample_size = 200) :
  seniors_in_sample school = 50 := by
  sorry

end NUMINAMATH_CALUDE_seniors_in_sample_is_fifty_l2109_210949


namespace NUMINAMATH_CALUDE_function_with_same_length_image_l2109_210929

-- Define the property for f
def HasSameLengthImage (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), a < b → ∃ (c d : ℝ), c < d ∧ 
    (Set.Ioo c d = f '' Set.Ioo a b) ∧ 
    (d - c = b - a)

-- State the theorem
theorem function_with_same_length_image (f : ℝ → ℝ) 
  (h : HasSameLengthImage f) : 
  ∃ (C : ℝ), (∀ x, f x = x + C) ∨ (∀ x, f x = -x + C) := by
  sorry

end NUMINAMATH_CALUDE_function_with_same_length_image_l2109_210929


namespace NUMINAMATH_CALUDE_ends_with_k_zeros_l2109_210908

/-- A p-adic integer with a nonzero last digit -/
def NonZeroLastDigitPAdicInteger (p : ℕ) (a : ℕ) : Prop :=
  Nat.Prime p ∧ a % p ≠ 0

theorem ends_with_k_zeros (p k : ℕ) (a : ℕ) 
  (h_p : Nat.Prime p) 
  (h_a : NonZeroLastDigitPAdicInteger p a) 
  (h_k : k > 0) :
  (a^(p^(k-1) * (p-1)) - 1) % p^k = 0 := by
sorry

end NUMINAMATH_CALUDE_ends_with_k_zeros_l2109_210908


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2109_210900

theorem rectangular_prism_volume
  (face_area1 face_area2 face_area3 : ℝ)
  (h1 : face_area1 = 15)
  (h2 : face_area2 = 20)
  (h3 : face_area3 = 30)
  (h4 : ∃ l w h : ℝ, l * w = face_area1 ∧ w * h = face_area2 ∧ l * h = face_area3) :
  ∃ volume : ℝ, volume = 30 * Real.sqrt 10 ∧
    (∀ l w h : ℝ, l * w = face_area1 → w * h = face_area2 → l * h = face_area3 →
      volume = l * w * h) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2109_210900


namespace NUMINAMATH_CALUDE_min_value_theorem_l2109_210956

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b) * b * c = 5) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y) * y * z = 5 → 2 * a + b + c ≤ 2 * x + y + z ∧
  2 * a + b + c = 2 * Real.sqrt 5 :=
by sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l2109_210956


namespace NUMINAMATH_CALUDE_two_days_saved_l2109_210934

/-- Represents the work scenario with original and additional workers --/
structure WorkScenario where
  originalMen : ℕ
  originalDays : ℕ
  additionalMen : ℕ
  totalWork : ℕ

/-- Calculates the number of days saved when additional workers join --/
def daysSaved (w : WorkScenario) : ℕ :=
  w.originalDays - (w.totalWork / (w.originalMen + w.additionalMen))

/-- Theorem stating that in the given scenario, 2 days are saved --/
theorem two_days_saved (w : WorkScenario) 
  (h1 : w.originalMen = 30)
  (h2 : w.originalDays = 8)
  (h3 : w.additionalMen = 10)
  (h4 : w.totalWork = w.originalMen * w.originalDays) :
  daysSaved w = 2 := by
  sorry

#eval daysSaved { originalMen := 30, originalDays := 8, additionalMen := 10, totalWork := 240 }

end NUMINAMATH_CALUDE_two_days_saved_l2109_210934


namespace NUMINAMATH_CALUDE_emelya_balls_count_l2109_210983

def total_balls : ℕ := 10
def broken_balls : ℕ := 3
def lost_balls : ℕ := 3

theorem emelya_balls_count :
  ∀ (M : ℝ),
  M > 0 →
  (broken_balls : ℝ) * M * (35/100) = (7/20) * M →
  ∃ (remaining_balls : ℕ),
  remaining_balls > 0 ∧
  (remaining_balls : ℝ) * M * (8/13) = (2/5) * M ∧
  total_balls = remaining_balls + broken_balls + lost_balls :=
by sorry

end NUMINAMATH_CALUDE_emelya_balls_count_l2109_210983


namespace NUMINAMATH_CALUDE_butterfly_failure_rate_l2109_210938

theorem butterfly_failure_rate 
  (total_caterpillars : ℕ) 
  (butterfly_price : ℚ) 
  (total_revenue : ℚ) : 
  total_caterpillars = 40 →
  butterfly_price = 3 →
  total_revenue = 72 →
  (total_caterpillars - (total_revenue / butterfly_price)) / total_caterpillars * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_failure_rate_l2109_210938


namespace NUMINAMATH_CALUDE_two_x_minus_y_value_l2109_210994

theorem two_x_minus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : x > y) :
  2 * x - y = 4 ∨ 2 * x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_two_x_minus_y_value_l2109_210994


namespace NUMINAMATH_CALUDE_book_sale_problem_l2109_210944

/-- Proves that the total cost of two books is 600, given the specified conditions --/
theorem book_sale_problem (cost_loss : ℝ) (selling_price : ℝ) :
  cost_loss = 350 →
  selling_price = cost_loss * (1 - 0.15) →
  ∃ (cost_gain : ℝ), 
    selling_price = cost_gain * (1 + 0.19) ∧
    cost_loss + cost_gain = 600 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_problem_l2109_210944


namespace NUMINAMATH_CALUDE_polynomial_must_be_constant_l2109_210911

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Sum of decimal digits of an integer's absolute value -/
def sumDecimalDigits (n : ℤ) : ℕ :=
  sorry

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Predicate for Fibonacci numbers -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

theorem polynomial_must_be_constant (P : IntPolynomial) :
  (∀ n : ℕ, n > 0 → ¬isFibonacci (sumDecimalDigits (P.eval n))) →
  P.degree = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_must_be_constant_l2109_210911


namespace NUMINAMATH_CALUDE_remainder_problem_l2109_210922

theorem remainder_problem : ∃ q : ℕ, 
  6598574241545098875458255622898854689448911257658451215825362549889 = 
  3721858987156557895464215545212524189541456658712589687354871258 * q + 8 * 23 + r ∧ 
  r < 23 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l2109_210922


namespace NUMINAMATH_CALUDE_monotonic_increase_intervals_l2109_210986

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem monotonic_increase_intervals (x : ℝ) :
  StrictMonoOn f (Set.Iio (-1)) ∧ StrictMonoOn f (Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_intervals_l2109_210986


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l2109_210939

-- Define s(n) as the sum of digits of n
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define the property that n - s(n) is divisible by 9
def divisible_by_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n - sum_of_digits n = 9 * k

-- State the theorem
theorem sum_of_digits_power_of_two :
  (∀ n : ℕ, divisible_by_nine n) →
  2^2009 % 9 = 5 →
  sum_of_digits (sum_of_digits (sum_of_digits (2^2009))) < 9 →
  sum_of_digits (sum_of_digits (sum_of_digits (2^2009))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l2109_210939


namespace NUMINAMATH_CALUDE_gcd_78_36_l2109_210914

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l2109_210914


namespace NUMINAMATH_CALUDE_connie_marbles_l2109_210905

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l2109_210905


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l2109_210913

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of possible four-ring arrangements on four fingers of one hand,
    given seven distinguishable rings, where the order matters and not all
    fingers need to have a ring -/
def ring_arrangements : ℕ :=
  choose 7 4 * factorial 4 * choose 7 3

theorem ring_arrangements_count :
  ring_arrangements = 29400 := by sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l2109_210913


namespace NUMINAMATH_CALUDE_negation_of_existence_l2109_210902

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > 3 - x₀) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≤ 3 - x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2109_210902


namespace NUMINAMATH_CALUDE_bag_pieces_problem_l2109_210951

theorem bag_pieces_problem (w b n : ℕ) : 
  b = 2 * w →                 -- The number of black pieces is twice the number of white pieces
  w - 2 * n = 1 →             -- After n rounds, 1 white piece is left
  b - 3 * n = 31 →            -- After n rounds, 31 black pieces are left
  b = 118 :=                  -- The initial number of black pieces was 118
by sorry

end NUMINAMATH_CALUDE_bag_pieces_problem_l2109_210951


namespace NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l2109_210903

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses Jessica added to the vase -/
def added_roses : ℕ := 16

/-- The final number of roses in the vase -/
def final_roses : ℕ := 23

/-- Theorem stating that the initial number of roses plus the added roses equals the final number of roses -/
theorem roses_equation : initial_roses + added_roses = final_roses := by sorry

/-- Theorem proving that the initial number of roses is 7 -/
theorem initial_roses_count : initial_roses = 7 := by sorry

end NUMINAMATH_CALUDE_roses_equation_initial_roses_count_l2109_210903


namespace NUMINAMATH_CALUDE_floor_abs_calculation_l2109_210926

theorem floor_abs_calculation : (((⌊|(-7.6 : ℝ)|⌋ : ℤ) + |⌊(-7.6 : ℝ)⌋|) : ℤ) * 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_calculation_l2109_210926


namespace NUMINAMATH_CALUDE_product_105_95_l2109_210975

theorem product_105_95 : 105 * 95 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_product_105_95_l2109_210975


namespace NUMINAMATH_CALUDE_binomial_expectation_variance_l2109_210915

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  X : ℝ → ℝ
  prob_zero : ℝ
  is_binomial : Prop
  prob_zero_eq : prob_zero = 1/3

/-- The expectation of a random variable -/
noncomputable def expectation (X : ℝ → ℝ) : ℝ := sorry

/-- The variance of a random variable -/
noncomputable def variance (X : ℝ → ℝ) : ℝ := sorry

theorem binomial_expectation_variance 
  (rv : BinomialRV) : 
  expectation (fun x => 3 * rv.X x + 2) = 4 ∧ 
  variance (fun x => 3 * rv.X x + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_variance_l2109_210915


namespace NUMINAMATH_CALUDE_pants_price_is_54_l2109_210962

/-- The price of a pair of pants Laura bought -/
def price_of_pants : ℕ := sorry

/-- The number of pairs of pants Laura bought -/
def num_pants : ℕ := 2

/-- The number of shirts Laura bought -/
def num_shirts : ℕ := 4

/-- The price of each shirt -/
def price_of_shirt : ℕ := 33

/-- The amount Laura gave to the cashier -/
def amount_given : ℕ := 250

/-- The change Laura received -/
def change_received : ℕ := 10

theorem pants_price_is_54 : price_of_pants = 54 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_is_54_l2109_210962


namespace NUMINAMATH_CALUDE_range_of_m_for_always_solvable_equation_l2109_210948

theorem range_of_m_for_always_solvable_equation :
  (∀ m : ℝ, ∃ x : ℝ, 4 * Real.cos x + Real.sin x ^ 2 + m - 4 = 0) →
  (∀ m : ℝ, m ∈ Set.Icc 0 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_always_solvable_equation_l2109_210948


namespace NUMINAMATH_CALUDE_magic_square_x_value_l2109_210906

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  sum_eq : a + b + c = d + e + f ∧ 
           a + b + c = g + h + i ∧ 
           a + b + c = a + d + g ∧ 
           a + b + c = b + e + h ∧ 
           a + b + c = c + f + i ∧ 
           a + b + c = a + e + i ∧ 
           a + b + c = c + e + g

/-- Theorem stating that x must be 230 in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a = x)
  (h2 : ms.b = 25)
  (h3 : ms.c = 110)
  (h4 : ms.d = 5) :
  x = 230 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l2109_210906


namespace NUMINAMATH_CALUDE_total_selling_price_proof_l2109_210909

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the number of toys whose cost price equals the total gain. -/
def totalSellingPrice (numToysSold : ℕ) (costPricePerToy : ℕ) (numToysGain : ℕ) : ℕ :=
  numToysSold * costPricePerToy + numToysGain * costPricePerToy

/-- Proves that the total selling price of 18 toys is 16800,
    given a cost price of 800 per toy and a gain equal to the cost of 3 toys. -/
theorem total_selling_price_proof :
  totalSellingPrice 18 800 3 = 16800 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_proof_l2109_210909


namespace NUMINAMATH_CALUDE_jack_emails_afternoon_l2109_210985

theorem jack_emails_afternoon (morning_emails : ℕ) (total_emails : ℕ) (afternoon_emails : ℕ) 
  (h1 : morning_emails = 3)
  (h2 : total_emails = 8)
  (h3 : afternoon_emails = total_emails - morning_emails) :
  afternoon_emails = 5 := by
  sorry

end NUMINAMATH_CALUDE_jack_emails_afternoon_l2109_210985


namespace NUMINAMATH_CALUDE_distance_run_in_two_hours_l2109_210998

/-- Given a person's running capabilities, calculate the distance they can run in 2 hours -/
theorem distance_run_in_two_hours 
  (distance : ℝ) -- The unknown distance the person can run in 2 hours
  (time_for_distance : ℝ) -- Time taken to run the unknown distance
  (track_length : ℝ) -- Length of the track
  (time_for_track : ℝ) -- Time taken to run the track
  (h1 : time_for_distance = 2) -- The person can run the unknown distance in 2 hours
  (h2 : track_length = 10000) -- The track is 10000 meters long
  (h3 : time_for_track = 10) -- It takes 10 hours to run the track
  (h4 : distance / time_for_distance = track_length / time_for_track) -- The speed is constant
  : distance = 2000 := by
  sorry

end NUMINAMATH_CALUDE_distance_run_in_two_hours_l2109_210998


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2109_210990

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2109_210990


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2109_210976

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2109_210976


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2109_210921

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 7 = a) :
  (a - (2*a - 1) / a) / ((a - 1) / a^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2109_210921


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2109_210950

theorem polynomial_coefficient_sum : 
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2109_210950


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2109_210989

/-- The polynomial f(x) = x^3 - 5x^2 + 8x - 4 -/
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 10*x + 8

theorem roots_of_polynomial :
  (f 1 = 0) ∧ 
  (f 2 = 0) ∧ 
  (f' 2 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 2) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2109_210989


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2109_210904

-- Define the functions
def f (a b x : ℝ) : ℝ := -2 * abs (x - a) + b
def g (c d x : ℝ) : ℝ := 2 * abs (x - c) + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) : 
  (f a b 1 = 4 ∧ f a b 7 = 0 ∧ g c d 1 = 4 ∧ g c d 7 = 0) → a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2109_210904


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l2109_210991

theorem max_value_cos_sin (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l2109_210991


namespace NUMINAMATH_CALUDE_theater_hall_seats_l2109_210961

/-- Represents a theater hall with three categories of seats. -/
structure TheaterHall where
  totalSeats : ℕ
  categoryIPrice : ℕ
  categoryIIPrice : ℕ
  categoryIIIPrice : ℕ
  freeTickets : ℕ
  revenueDifference : ℕ

/-- Checks if the theater hall satisfies all given conditions. -/
def validTheaterHall (hall : TheaterHall) : Prop :=
  (hall.totalSeats % 5 = 0) ∧
  (hall.categoryIPrice = 220) ∧
  (hall.categoryIIPrice = 200) ∧
  (hall.categoryIIIPrice = 180) ∧
  (hall.freeTickets = 150) ∧
  (hall.revenueDifference = 4320)

/-- Theorem stating that a valid theater hall has 360 seats. -/
theorem theater_hall_seats (hall : TheaterHall) 
  (h : validTheaterHall hall) : hall.totalSeats = 360 := by
  sorry

#check theater_hall_seats

end NUMINAMATH_CALUDE_theater_hall_seats_l2109_210961


namespace NUMINAMATH_CALUDE_smallest_sum_20_consecutive_twice_square_l2109_210960

/-- The sum of 20 consecutive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- Predicate to check if a number is twice a perfect square -/
def is_twice_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = 2 * k^2

/-- The smallest sum of 20 consecutive positive integers that is twice a perfect square -/
theorem smallest_sum_20_consecutive_twice_square : 
  (∃ n : ℕ, sum_20_consecutive n = 450 ∧ 
    is_twice_perfect_square (sum_20_consecutive n) ∧
    ∀ m : ℕ, m < n → ¬(is_twice_perfect_square (sum_20_consecutive m))) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_20_consecutive_twice_square_l2109_210960


namespace NUMINAMATH_CALUDE_union_complement_eq_set_l2109_210965

universe u

def U : Finset ℕ := {1,2,3,4,5}
def M : Finset ℕ := {1,4}
def N : Finset ℕ := {2,5}

theorem union_complement_eq_set : N ∪ (U \ M) = {2,3,5} := by sorry

end NUMINAMATH_CALUDE_union_complement_eq_set_l2109_210965


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2109_210933

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a*b + b*c + a*c = 100) : 
  a + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2109_210933


namespace NUMINAMATH_CALUDE_calculator_problem_l2109_210981

theorem calculator_problem (x : ℝ) (hx : x ≠ 0) :
  (1 / (1/x - 1)) - 1 = -0.75 → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_calculator_problem_l2109_210981


namespace NUMINAMATH_CALUDE_complex_magnitude_l2109_210943

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2109_210943


namespace NUMINAMATH_CALUDE_uma_money_fraction_l2109_210935

theorem uma_money_fraction (rita sam tina unknown : ℚ) : 
  rita > 0 ∧ sam > 0 ∧ tina > 0 ∧ unknown > 0 →
  rita / 6 = sam / 5 ∧ rita / 6 = tina / 7 ∧ rita / 6 = unknown / 8 →
  (rita / 6 + sam / 5 + tina / 7 + unknown / 8) / (rita + sam + tina + unknown) = 2 / 13 := by
sorry

end NUMINAMATH_CALUDE_uma_money_fraction_l2109_210935


namespace NUMINAMATH_CALUDE_al_karhi_square_root_approximation_l2109_210941

theorem al_karhi_square_root_approximation 
  (N a r : ℝ) 
  (h1 : N > 0) 
  (h2 : a > 0) 
  (h3 : a^2 ≤ N) 
  (h4 : (a+1)^2 > N) 
  (h5 : r = N - a^2) 
  (h6 : r < 2*a + 1) : 
  ∃ (ε : ℝ), ε > 0 ∧ |Real.sqrt N - (a + r / (2*a + 1))| < ε :=
sorry

end NUMINAMATH_CALUDE_al_karhi_square_root_approximation_l2109_210941


namespace NUMINAMATH_CALUDE_special_rectangle_exists_l2109_210972

/-- A rectangle with the given properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_equals_area : 2 * (length + width) = length * width
  width_is_length_minus_three : width = length - 3

/-- The theorem stating that a rectangle with length 6 and width 3 satisfies the conditions --/
theorem special_rectangle_exists : ∃ (r : SpecialRectangle), r.length = 6 ∧ r.width = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_exists_l2109_210972
