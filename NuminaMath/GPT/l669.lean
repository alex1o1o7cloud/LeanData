import Mathlib

namespace NUMINAMATH_GPT_pet_store_initial_gerbils_l669_66954

-- Define sold gerbils
def sold_gerbils : ℕ := 69

-- Define left gerbils
def left_gerbils : ℕ := 16

-- Define the initial number of gerbils
def initial_gerbils : ℕ := sold_gerbils + left_gerbils

-- State the theorem to be proved
theorem pet_store_initial_gerbils : initial_gerbils = 85 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_pet_store_initial_gerbils_l669_66954


namespace NUMINAMATH_GPT_triangle_angle_bisectors_l669_66950

theorem triangle_angle_bisectors {a b c : ℝ} (ht : (a = 2 ∧ b = 3 ∧ c < 5)) : 
  (∃ h_a h_b h_c : ℝ, h_a + h_b > h_c ∧ h_a + h_c > h_b ∧ h_b + h_c > h_a) →
  ¬ (∃ ell_a ell_b ell_c : ℝ, ell_a + ell_b > ell_c ∧ ell_a + ell_c > ell_b ∧ ell_b + ell_c > ell_a) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_bisectors_l669_66950


namespace NUMINAMATH_GPT_third_pipe_empty_time_l669_66979

theorem third_pipe_empty_time :
  let A_rate := 1/60
  let B_rate := 1/75
  let combined_rate := 1/50
  let third_pipe_rate := combined_rate - (A_rate + B_rate)
  let time_to_empty := 1 / third_pipe_rate
  time_to_empty = 100 :=
by
  sorry

end NUMINAMATH_GPT_third_pipe_empty_time_l669_66979


namespace NUMINAMATH_GPT_bacteria_growth_returns_six_l669_66993

theorem bacteria_growth_returns_six (n : ℕ) (h : (4 * 2 ^ n > 200)) : n = 6 :=
sorry

end NUMINAMATH_GPT_bacteria_growth_returns_six_l669_66993


namespace NUMINAMATH_GPT_rulers_left_in_drawer_l669_66944

theorem rulers_left_in_drawer (initial_rulers taken_rulers : ℕ) (h1 : initial_rulers = 46) (h2 : taken_rulers = 25) :
  initial_rulers - taken_rulers = 21 :=
by
  sorry

end NUMINAMATH_GPT_rulers_left_in_drawer_l669_66944


namespace NUMINAMATH_GPT_smallest_integer_is_77_l669_66997

theorem smallest_integer_is_77 
  (A B C D E F G : ℤ)
  (h_uniq: A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G)
  (h_sum: A + B + C + D + E + F + G = 840)
  (h_largest: G = 190)
  (h_two_smallest_sum: A + B = 156) : 
  A = 77 :=
sorry

end NUMINAMATH_GPT_smallest_integer_is_77_l669_66997


namespace NUMINAMATH_GPT_find_z_l669_66987

-- Definitions of the conditions
def equation_1 (x y : ℝ) : Prop := x^2 - 3 * x + 6 = y - 10
def equation_2 (y z : ℝ) : Prop := y = 2 * z
def x_value (x : ℝ) : Prop := x = -5

-- Lean theorem statement
theorem find_z (x y z : ℝ) (h1 : equation_1 x y) (h2 : equation_2 y z) (h3 : x_value x) : z = 28 :=
sorry

end NUMINAMATH_GPT_find_z_l669_66987


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l669_66978

noncomputable def sequence_increasing_condition (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) > |a n|

noncomputable def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n < a (n + 1)

theorem sufficient_not_necessary_condition (a : ℕ → ℝ) :
  sequence_increasing_condition a → is_increasing_sequence a ∧ ¬(∀ b : ℕ → ℝ, is_increasing_sequence b → sequence_increasing_condition b) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l669_66978


namespace NUMINAMATH_GPT_winning_configurations_for_blake_l669_66999

def isWinningConfigurationForBlake (config : List ℕ) := 
  let nimSum := config.foldl (xor) 0
  nimSum = 0

theorem winning_configurations_for_blake :
  (isWinningConfigurationForBlake [8, 2, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 3, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 5, 2]) :=
by {
  sorry
}

end NUMINAMATH_GPT_winning_configurations_for_blake_l669_66999


namespace NUMINAMATH_GPT_jeremy_home_to_school_distance_l669_66918

theorem jeremy_home_to_school_distance (v d : ℝ) (h1 : 30 / 60 = 1 / 2) (h2 : 15 / 60 = 1 / 4)
  (h3 : d = v * (1 / 2)) (h4 : d = (v + 12) * (1 / 4)):
  d = 6 :=
by
  -- We assume that the conditions given lead to the distance being 6 miles
  sorry

end NUMINAMATH_GPT_jeremy_home_to_school_distance_l669_66918


namespace NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l669_66927

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l669_66927


namespace NUMINAMATH_GPT_interval_of_decrease_for_f_l669_66962

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3)

def decreasing_interval (s : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem interval_of_decrease_for_f :
  decreasing_interval {x : ℝ | x < -1} f :=
by
  sorry

end NUMINAMATH_GPT_interval_of_decrease_for_f_l669_66962


namespace NUMINAMATH_GPT_greatest_cars_with_ac_not_racing_stripes_l669_66973

def total_cars : ℕ := 100
def without_ac : ℕ := 49
def at_least_racing_stripes : ℕ := 51

theorem greatest_cars_with_ac_not_racing_stripes :
  (total_cars - without_ac) - (at_least_racing_stripes - without_ac) = 49 :=
by
  unfold total_cars without_ac at_least_racing_stripes
  sorry

end NUMINAMATH_GPT_greatest_cars_with_ac_not_racing_stripes_l669_66973


namespace NUMINAMATH_GPT_masha_mushrooms_l669_66925

theorem masha_mushrooms (B1 B2 B3 B4 G1 G2 G3 : ℕ) (total : B1 + B2 + B3 + B4 + G1 + G2 + G3 = 70)
  (girls_distinct : G1 ≠ G2 ∧ G1 ≠ G3 ∧ G2 ≠ G3)
  (boys_threshold : ∀ {A B C D : ℕ}, (A = B1 ∨ A = B2 ∨ A = B3 ∨ A = B4) →
                    (B = B1 ∨ B = B2 ∨ B = B3 ∨ B = B4) →
                    (C = B1 ∨ C = B2 ∨ C = B3 ∨ C = B4) → 
                    (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
                    A + B + C ≥ 43)
  (diff_no_more_than_five_times : ∀ {x y : ℕ}, (x = B1 ∨ x = B2 ∨ x = B3 ∨ x = B4 ∨ x = G1 ∨ x = G2 ∨ x = G3) →
                                  (y = B1 ∨ y = B2 ∨ y = B3 ∨ y = B4 ∨ y = G1 ∨ y = G2 ∨ y = G3) →
                                  x ≠ y → x ≤ 5 * y ∧ y ≤ 5 * x)
  (masha_max_girl : G3 = max G1 (max G2 G3))
  : G3 = 5 :=
sorry

end NUMINAMATH_GPT_masha_mushrooms_l669_66925


namespace NUMINAMATH_GPT_find_least_skilled_painter_l669_66914

-- Define the genders
inductive Gender
| Male
| Female

-- Define the family members
inductive Member
| Grandmother
| Niece
| Nephew
| Granddaughter

-- Define a structure to hold the properties of each family member
structure Properties where
  gender : Gender
  age : Nat
  isTwin : Bool

-- Assume the properties of each family member as given
def grandmother : Properties := { gender := Gender.Female, age := 70, isTwin := false }
def niece : Properties := { gender := Gender.Female, age := 20, isTwin := false }
def nephew : Properties := { gender := Gender.Male, age := 20, isTwin := true }
def granddaughter : Properties := { gender := Gender.Female, age := 20, isTwin := true }

-- Define the best painter
def bestPainter := niece

-- Conditions based on the problem (rephrased to match formalization)
def conditions (least_skilled : Member) : Prop :=
  (bestPainter.gender ≠ (match least_skilled with
                          | Member.Grandmother => grandmother
                          | Member.Niece => niece
                          | Member.Nephew => nephew
                          | Member.Granddaughter => granddaughter ).gender) ∧
  ((match least_skilled with
    | Member.Grandmother => grandmother
    | Member.Niece => niece
    | Member.Nephew => nephew
    | Member.Granddaughter => granddaughter ).isTwin) ∧
  (bestPainter.age = (match least_skilled with
                      | Member.Grandmother => grandmother
                      | Member.Niece => niece
                      | Member.Nephew => nephew
                      | Member.Granddaughter => granddaughter ).age)

-- Statement of the problem
theorem find_least_skilled_painter : ∃ m : Member, conditions m ∧ m = Member.Granddaughter :=
by
  sorry

end NUMINAMATH_GPT_find_least_skilled_painter_l669_66914


namespace NUMINAMATH_GPT_incorrect_inequality_l669_66977

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end NUMINAMATH_GPT_incorrect_inequality_l669_66977


namespace NUMINAMATH_GPT_diagonal_cubes_140_320_360_l669_66953

-- Define the problem parameters 
def length_x : ℕ := 140
def length_y : ℕ := 320
def length_z : ℕ := 360

-- Define the function to calculate the number of unit cubes the internal diagonal passes through.
def num_cubes_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - Nat.gcd x y - Nat.gcd y z - Nat.gcd z x + Nat.gcd (Nat.gcd x y) z

-- The target theorem to be proven
theorem diagonal_cubes_140_320_360 :
  num_cubes_diagonal length_x length_y length_z = 760 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_cubes_140_320_360_l669_66953


namespace NUMINAMATH_GPT_angle_complement_supplement_l669_66923

theorem angle_complement_supplement (x : ℝ) (h1 : 90 - x = (1 / 2) * (180 - x)) : x = 90 := by
  sorry

end NUMINAMATH_GPT_angle_complement_supplement_l669_66923


namespace NUMINAMATH_GPT_needed_correct_to_pass_l669_66972

def total_questions : Nat := 120
def genetics_questions : Nat := 20
def ecology_questions : Nat := 50
def evolution_questions : Nat := 50

def correct_genetics : Nat := (60 * genetics_questions) / 100
def correct_ecology : Nat := (50 * ecology_questions) / 100
def correct_evolution : Nat := (70 * evolution_questions) / 100
def total_correct : Nat := correct_genetics + correct_ecology + correct_evolution

def passing_rate : Nat := 65
def passing_score : Nat := (passing_rate * total_questions) / 100

theorem needed_correct_to_pass : (passing_score - total_correct) = 6 := 
by
  sorry

end NUMINAMATH_GPT_needed_correct_to_pass_l669_66972


namespace NUMINAMATH_GPT_find_x_l669_66966

def set_of_numbers := [1, 2, 4, 5, 6, 9, 9, 10]

theorem find_x {x : ℝ} (h : (set_of_numbers.sum + x) / 9 = 7) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l669_66966


namespace NUMINAMATH_GPT_initial_tickets_l669_66948

-- Definitions of the conditions
def ferris_wheel_rides : ℕ := 2
def roller_coaster_rides : ℕ := 3
def log_ride_rides : ℕ := 7

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 1

def additional_tickets_needed : ℕ := 6

-- Calculate the total number of tickets needed
def total_tickets_needed : ℕ := 
  (ferris_wheel_rides * ferris_wheel_cost) +
  (roller_coaster_rides * roller_coaster_cost) +
  (log_ride_rides * log_ride_cost)

-- The proof statement
theorem initial_tickets : ∀ (initial_tickets : ℕ), 
  total_tickets_needed - additional_tickets_needed = initial_tickets → 
  initial_tickets = 20 :=
by
  intros initial_tickets h
  sorry

end NUMINAMATH_GPT_initial_tickets_l669_66948


namespace NUMINAMATH_GPT_factorization_correct_l669_66904

theorem factorization_correct (x y : ℝ) : 
  x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := 
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l669_66904


namespace NUMINAMATH_GPT_rectangle_length_width_difference_l669_66937

theorem rectangle_length_width_difference :
  ∃ (length width : ℕ), (length * width = 864) ∧ (length + width = 60) ∧ (length - width = 12) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_width_difference_l669_66937


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l669_66965

noncomputable def is_perpendicular (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

theorem perpendicular_line_through_point
  (line : ℝ → ℝ)
  (P : ℝ × ℝ)
  (h_line_eq : ∀ x, line x = 3 * x + 8)
  (hP : P = (2,1)) :
  ∃ a b c : ℝ, a * (P.1) + b * (P.2) + c = 0 ∧ is_perpendicular 3 (-b / a) ∧ a * 1 + b * 3 + c = 0 :=
sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l669_66965


namespace NUMINAMATH_GPT_number_of_students_l669_66940

theorem number_of_students 
  (n : ℕ)
  (h1: 108 - 36 = 72)
  (h2: ∀ n > 0, 108 / n - 72 / n = 3) :
  n = 12 :=
sorry

end NUMINAMATH_GPT_number_of_students_l669_66940


namespace NUMINAMATH_GPT_Mikail_money_left_after_purchase_l669_66968

def Mikail_age_tomorrow : ℕ := 9  -- Defining Mikail's age tomorrow as 9.

def gift_per_year : ℕ := 5  -- Defining the gift amount per year of age as $5.

def video_game_cost : ℕ := 80  -- Defining the cost of the video game as $80.

def calculate_gift (age : ℕ) : ℕ := age * gift_per_year  -- Function to calculate the gift money he receives based on his age.

-- The statement we need to prove:
theorem Mikail_money_left_after_purchase : 
    calculate_gift Mikail_age_tomorrow < video_game_cost → calculate_gift Mikail_age_tomorrow - video_game_cost = 0 :=
by
  sorry

end NUMINAMATH_GPT_Mikail_money_left_after_purchase_l669_66968


namespace NUMINAMATH_GPT_sum_of_squares_of_projections_constant_l669_66985

-- Define the sum of the squares of projections function
noncomputable def sum_of_squares_of_projections (a : ℝ) (α : ℝ) : ℝ :=
  let p1 := a * Real.cos α
  let p2 := a * Real.cos (Real.pi / 3 - α)
  let p3 := a * Real.cos (Real.pi / 3 + α)
  p1^2 + p2^2 + p3^2

-- Statement of the theorem
theorem sum_of_squares_of_projections_constant (a α : ℝ) : 
  sum_of_squares_of_projections a α = 3 / 2 * a^2 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_projections_constant_l669_66985


namespace NUMINAMATH_GPT_tangent_line_at_neg1_l669_66901

-- Define the function given in the condition.
def f (x : ℝ) : ℝ := x^2 + 4 * x + 2

-- Define the point of tangency given in the condition.
def point_of_tangency : ℝ × ℝ := (-1, f (-1))

-- Define the derivative of the function.
def derivative_f (x : ℝ) : ℝ := 2 * x + 4

-- The proof statement: the equation of the tangent line at x = -1 is y = 2x + 1
theorem tangent_line_at_neg1 :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = f x → derivative_f (-1) = m ∧ point_of_tangency.fst = -1 ∧ y = m * (x + 1) + b) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_neg1_l669_66901


namespace NUMINAMATH_GPT_card_draw_probability_l669_66931

theorem card_draw_probability:
  let hearts := 13
  let diamonds := 13
  let clubs := 13
  let total_cards := 52
  let first_draw_probability := hearts / (total_cards : ℝ)
  let second_draw_probability := diamonds / (total_cards - 1 : ℝ)
  let third_draw_probability := clubs / (total_cards - 2 : ℝ)
  first_draw_probability * second_draw_probability * third_draw_probability = 2197 / 132600 :=
by
  sorry

end NUMINAMATH_GPT_card_draw_probability_l669_66931


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l669_66921

-- Problem 1
theorem simplify_expression1 (x y : ℤ) :
  (-3) * x + 2 * y - 5 * x - 7 * y = -8 * x - 5 * y :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℤ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l669_66921


namespace NUMINAMATH_GPT_equation_of_plane_l669_66920

/--
The equation of the plane passing through the points (2, -2, 2) and (0, 0, 2),
and which is perpendicular to the plane 2x - y + 4z = 8, is given by:
Ax + By + Cz + D = 0 where A, B, C, D are integers such that A > 0 and gcd(|A|,|B|,|C|,|D|) = 1.
-/
theorem equation_of_plane :
  ∃ (A B C D : ℤ),
    A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
    (∀ x y z : ℤ, A * x + B * y + C * z + D = 0 ↔ x + y = 0) :=
sorry

end NUMINAMATH_GPT_equation_of_plane_l669_66920


namespace NUMINAMATH_GPT_two_squares_always_similar_l669_66996

-- Define geometric shapes and their properties
inductive Shape
| Rectangle : Shape
| Rhombus   : Shape
| Square    : Shape
| RightAngledTriangle : Shape

-- Define similarity condition
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.Square, Shape.Square => true
  | _, _ => false

-- Prove that two squares are always similar
theorem two_squares_always_similar : similar Shape.Square Shape.Square = true :=
by
  sorry

end NUMINAMATH_GPT_two_squares_always_similar_l669_66996


namespace NUMINAMATH_GPT_mustard_at_first_table_l669_66992

theorem mustard_at_first_table (M : ℝ) :
  (M + 0.25 + 0.38 = 0.88) → M = 0.25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mustard_at_first_table_l669_66992


namespace NUMINAMATH_GPT_captain_age_is_24_l669_66912

theorem captain_age_is_24 (C W : ℕ) 
  (hW : W = C + 7)
  (h_total_team_age : 23 * 11 = 253)
  (h_total_9_players_age : 22 * 9 = 198)
  (h_team_age_equation : 253 = 198 + C + W)
  : C = 24 :=
sorry

end NUMINAMATH_GPT_captain_age_is_24_l669_66912


namespace NUMINAMATH_GPT_f_2009_l669_66930

def f (x : ℝ) : ℝ := x^3 -- initial definition for x in [-1, 1]

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom symmetric_around_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_cubed : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3

theorem f_2009 : f 2009 = 1 := by {
  -- The body of the theorem will be filled with proof steps
  sorry
}

end NUMINAMATH_GPT_f_2009_l669_66930


namespace NUMINAMATH_GPT_odd_n_divides_pow_fact_sub_one_l669_66909

theorem odd_n_divides_pow_fact_sub_one
  {n : ℕ} (hn_pos : n > 0) (hn_odd : n % 2 = 1)
  : n ∣ (2 ^ (Nat.factorial n) - 1) :=
sorry

end NUMINAMATH_GPT_odd_n_divides_pow_fact_sub_one_l669_66909


namespace NUMINAMATH_GPT_determine_n_l669_66949

theorem determine_n (n : ℕ) (h1 : 0 < n) 
(h2 : ∃ (sols : Finset (ℕ × ℕ × ℕ)), 
  (∀ (x y z : ℕ), (x, y, z) ∈ sols ↔ 3 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ sols.card = 55) : 
  n = 36 := 
by 
  sorry 

end NUMINAMATH_GPT_determine_n_l669_66949


namespace NUMINAMATH_GPT_no_real_a_b_l669_66960

noncomputable def SetA (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

noncomputable def SetB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

noncomputable def SetC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 144}

theorem no_real_a_b :
  ¬ ∃ (a b : ℝ), (∃ p ∈ SetA a b, p ∈ SetB) ∧ (a, b) ∈ SetC :=
by
    sorry

end NUMINAMATH_GPT_no_real_a_b_l669_66960


namespace NUMINAMATH_GPT_lines_are_coplanar_l669_66906

/- Define the parameterized lines -/
def L1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - k * s, 2 + 2 * k * s)
def L2 (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 7 + 3 * t, 1 - 2 * t)

/- Prove that k = 0 ensures the lines are coplanar -/
theorem lines_are_coplanar (k : ℝ) : k = 0 ↔ 
  ∃ (s t : ℝ), L1 s k = L2 t :=
by {
  sorry
}

end NUMINAMATH_GPT_lines_are_coplanar_l669_66906


namespace NUMINAMATH_GPT_part_one_part_two_l669_66974

-- Defining the function and its first derivative
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Part (Ⅰ)
theorem part_one (a b : ℝ)
  (H1 : f' a b 3 = 24)
  (H2 : f' a b 1 = 0) :
  a = 1 ∧ b = -3 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1 → f' 1 (-3) x ≤ 0) :=
sorry

-- Part (Ⅱ)
theorem part_two (b : ℝ)
  (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 3 * x^2 + b ≤ 0) :
  b ≤ -3 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l669_66974


namespace NUMINAMATH_GPT_chocolate_cost_in_promotion_l669_66938

/-!
Bernie buys two chocolates every week at a local store, where one chocolate costs $3.
In a different store with a promotion, each chocolate costs some amount and Bernie would save $6 
in three weeks if he bought his chocolates there. Prove that the cost of one chocolate 
in the store with the promotion is $2.
-/

theorem chocolate_cost_in_promotion {n p_local savings : ℕ} (weeks : ℕ) (p_promo : ℕ)
  (h_n : n = 2)
  (h_local : p_local = 3)
  (h_savings : savings = 6)
  (h_weeks : weeks = 3)
  (h_promo : p_promo = (p_local * n * weeks - savings) / (n * weeks)) :
  p_promo = 2 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_chocolate_cost_in_promotion_l669_66938


namespace NUMINAMATH_GPT_evaluate_dollar_l669_66939

variable {R : Type} [CommRing R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : 
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_dollar_l669_66939


namespace NUMINAMATH_GPT_max_cookie_price_l669_66991

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end NUMINAMATH_GPT_max_cookie_price_l669_66991


namespace NUMINAMATH_GPT_xiao_gang_steps_l669_66926

theorem xiao_gang_steps (x : ℕ) (H1 : 9000 / x = 13500 / (x + 15)) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_xiao_gang_steps_l669_66926


namespace NUMINAMATH_GPT_magic_square_sum_l669_66900

variable {a b c d e : ℕ}

-- Given conditions:
-- It's a magic square and the sums of the numbers in each row, column, and diagonal are equal.
-- Positions and known values specified:
theorem magic_square_sum (h : 15 + 24 = 18 + c ∧ 18 + c = 27 + a ∧ c = 21 ∧ a = 12 ∧ e = 17 ∧ d = 30 ∧ b = 25)
: d + e = 47 :=
by
  -- Sorry used to skip the proof
  sorry

end NUMINAMATH_GPT_magic_square_sum_l669_66900


namespace NUMINAMATH_GPT_number_of_valid_selections_l669_66924

theorem number_of_valid_selections : 
  ∃ combinations : Finset (Finset ℕ), 
    combinations = {
      {2, 6, 3, 5}, 
      {2, 6, 1, 7}, 
      {2, 4, 1, 5}, 
      {4, 1, 3}, 
      {6, 1, 5}, 
      {4, 6, 3, 7}, 
      {2, 4, 6, 5, 7}
    } ∧ combinations.card = 7 :=
by sorry

end NUMINAMATH_GPT_number_of_valid_selections_l669_66924


namespace NUMINAMATH_GPT_optimal_messenger_strategy_l669_66955

theorem optimal_messenger_strategy (p : ℝ) (hp : 0 < p ∧ p < 1) :
  (p < 1/3 → ∃ n : ℕ, n = 4 ∧ ∀ (k : ℕ), k = 10) ∧ 
  (1/3 ≤ p → ∃ n : ℕ, n = 2 ∧ ∀ (m : ℕ), m = 20) :=
by
  sorry

end NUMINAMATH_GPT_optimal_messenger_strategy_l669_66955


namespace NUMINAMATH_GPT_largest_consecutive_sum_is_nine_l669_66919

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end NUMINAMATH_GPT_largest_consecutive_sum_is_nine_l669_66919


namespace NUMINAMATH_GPT_part1_l669_66907

theorem part1 (a b c d : ℤ) (h : a * d - b * c = 1) : Int.gcd (a + b) (c + d) = 1 :=
sorry

end NUMINAMATH_GPT_part1_l669_66907


namespace NUMINAMATH_GPT_positive_number_square_sum_eq_210_l669_66934

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end NUMINAMATH_GPT_positive_number_square_sum_eq_210_l669_66934


namespace NUMINAMATH_GPT_total_jellybeans_l669_66915

def nephews := 3
def nieces := 2
def jellybeans_per_child := 14
def children := nephews + nieces

theorem total_jellybeans : children * jellybeans_per_child = 70 := by
  sorry

end NUMINAMATH_GPT_total_jellybeans_l669_66915


namespace NUMINAMATH_GPT_calculate_binom_l669_66958

theorem calculate_binom : 2 * Nat.choose 30 3 = 8120 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_binom_l669_66958


namespace NUMINAMATH_GPT_ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l669_66951

-- Define the basic conditions of the figures
def regular_pentagon (side_length : ℕ) : ℝ := 5 * side_length

-- Define ink length of a figure n
def ink_length (n : ℕ) : ℝ :=
  if n = 1 then regular_pentagon 1 else
  regular_pentagon (n-1) + (3 * (n - 1) + 2)

-- Part (a): Ink length of Figure 4
theorem ink_length_figure_4 : ink_length 4 = 38 := 
  by sorry

-- Part (b): Difference between ink length of Figure 9 and Figure 8
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 29 :=
  by sorry

-- Part (c): Ink length of Figure 100
theorem ink_length_figure_100 : ink_length 100 = 15350 :=
  by sorry

end NUMINAMATH_GPT_ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l669_66951


namespace NUMINAMATH_GPT_total_salmon_count_l669_66943

def chinook_males := 451228
def chinook_females := 164225
def sockeye_males := 212001
def sockeye_females := 76914
def coho_males := 301008
def coho_females := 111873
def pink_males := 518001
def pink_females := 182945
def chum_males := 230023
def chum_females := 81321

theorem total_salmon_count : 
  chinook_males + chinook_females + 
  sockeye_males + sockeye_females + 
  coho_males + coho_females + 
  pink_males + pink_females + 
  chum_males + chum_females = 2329539 := 
by
  sorry

end NUMINAMATH_GPT_total_salmon_count_l669_66943


namespace NUMINAMATH_GPT_sequence_a4_value_l669_66976

theorem sequence_a4_value : 
  ∀ (a : ℕ → ℕ), a 1 = 2 → (∀ n, n ≥ 2 → a n = a (n - 1) + n) → a 4 = 11 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a4_value_l669_66976


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l669_66984

theorem arithmetic_sequence_a7 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 4 = 9) 
  (common_diff : ∀ n, a (n + 1) = a n + d) :
  a 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l669_66984


namespace NUMINAMATH_GPT_tea_in_box_l669_66969

theorem tea_in_box (tea_per_day ounces_per_week ounces_per_box : ℝ) 
    (H1 : tea_per_day = 1 / 5) 
    (H2 : ounces_per_week = tea_per_day * 7) 
    (H3 : ounces_per_box = ounces_per_week * 20) : 
    ounces_per_box = 28 := 
by
  sorry

end NUMINAMATH_GPT_tea_in_box_l669_66969


namespace NUMINAMATH_GPT_pieces_length_l669_66970

theorem pieces_length (L M S : ℝ) (h1 : L + M + S = 180)
  (h2 : L = M + S + 30)
  (h3 : M = L / 2 - 10) :
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_pieces_length_l669_66970


namespace NUMINAMATH_GPT_trapezoid_perimeter_calc_l669_66928

theorem trapezoid_perimeter_calc 
  (EF GH : ℝ) (d : ℝ)
  (h_parallel : EF = 10) 
  (h_eq : GH = 22) 
  (h_distance : d = 5) 
  (h_parallel_cond : EF = 10 ∧ GH = 22 ∧ d = 5) 
: 32 + 2 * Real.sqrt 61 = (10 : ℝ) + 2 * (Real.sqrt ((12 / 2)^2 + 5^2)) + 22 := 
by {
  -- The proof goes here, but for now it's omitted
  sorry
}

end NUMINAMATH_GPT_trapezoid_perimeter_calc_l669_66928


namespace NUMINAMATH_GPT_functional_equality_l669_66936

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equality
  (h1 : ∀ x : ℝ, f x ≤ x)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_functional_equality_l669_66936


namespace NUMINAMATH_GPT_base8_to_base10_conversion_l669_66945

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end NUMINAMATH_GPT_base8_to_base10_conversion_l669_66945


namespace NUMINAMATH_GPT_green_pill_cost_l669_66942

-- Define the conditions 
variables (pinkCost greenCost : ℝ)
variable (totalCost : ℝ := 819) -- total cost for three weeks
variable (days : ℝ := 21) -- number of days in three weeks

-- Establish relationships between pink and green pill costs
axiom greenIsMore : greenCost = pinkCost + 1
axiom dailyCost : 2 * greenCost + pinkCost = 39

-- Define the theorem to prove the cost of one green pill
theorem green_pill_cost : greenCost = 40/3 :=
by
  -- Proof would go here, but is omitted for now.
  sorry

end NUMINAMATH_GPT_green_pill_cost_l669_66942


namespace NUMINAMATH_GPT_sampling_survey_suitability_l669_66981

-- Define the conditions
def OptionA := "Understanding the effectiveness of a certain drug"
def OptionB := "Understanding the vision status of students in this class"
def OptionC := "Organizing employees of a unit to undergo physical examinations at a hospital"
def OptionD := "Inspecting components of artificial satellite"

-- Mathematical statement
theorem sampling_survey_suitability : OptionA = "Understanding the effectiveness of a certain drug" → 
  ∃ (suitable_for_sampling_survey : String), suitable_for_sampling_survey = OptionA :=
by
  sorry

end NUMINAMATH_GPT_sampling_survey_suitability_l669_66981


namespace NUMINAMATH_GPT_both_hit_exactly_one_hits_at_least_one_hits_l669_66933

noncomputable def prob_A : ℝ := 0.8
noncomputable def prob_B : ℝ := 0.9

theorem both_hit : prob_A * prob_B = 0.72 := by
  sorry

theorem exactly_one_hits : prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = 0.26 := by
  sorry

theorem at_least_one_hits : 1 - (1 - prob_A) * (1 - prob_B) = 0.98 := by
  sorry

end NUMINAMATH_GPT_both_hit_exactly_one_hits_at_least_one_hits_l669_66933


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l669_66908

theorem repeating_decimal_fraction (x : ℚ) (h : x = 7.5656) : x = 749 / 99 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l669_66908


namespace NUMINAMATH_GPT_find_k_l669_66910

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

-- Given that the union of sets A and B is {1, 2, 3, 5}, prove that k = 3.
theorem find_k (k : ℕ) (h : A k ∪ B = {1, 2, 3, 5}) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l669_66910


namespace NUMINAMATH_GPT_valid_password_count_l669_66994

/-- 
The number of valid 4-digit ATM passwords at Fred's Bank, composed of digits from 0 to 9,
that do not start with the sequence "9,1,1" and do not end with the digit "5",
is 8991.
-/
theorem valid_password_count : 
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  total_passwords - (start_911 + end_5 - start_911_end_5) = 8991 :=
by
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  show total_passwords - (start_911 + end_5 - start_911_end_5) = 8991
  sorry

end NUMINAMATH_GPT_valid_password_count_l669_66994


namespace NUMINAMATH_GPT_expected_value_is_100_cents_l669_66952

-- Definitions for the values of the coins
def value_quarter : ℕ := 25
def value_half_dollar : ℕ := 50
def value_dollar : ℕ := 100

-- Define the total value of all coins
def total_value : ℕ := 2 * value_quarter + value_half_dollar + value_dollar

-- Probability of heads for a single coin
def p_heads : ℚ := 1 / 2

-- Expected value calculation
def expected_value : ℚ := p_heads * ↑total_value

-- The theorem we need to prove
theorem expected_value_is_100_cents : expected_value = 100 :=
by
  -- This is where the proof would go, but we are omitting it
  sorry

end NUMINAMATH_GPT_expected_value_is_100_cents_l669_66952


namespace NUMINAMATH_GPT_workshop_male_workers_l669_66941

variables (F M : ℕ)

theorem workshop_male_workers :
  (M = F + 45) ∧ (M - 5 = 3 * F) → M = 65 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_workshop_male_workers_l669_66941


namespace NUMINAMATH_GPT_sachin_age_l669_66946
-- Import the necessary library

-- Lean statement defining the problem conditions and result
theorem sachin_age :
  ∃ (S R : ℝ), (R = S + 7) ∧ (S / R = 7 / 9) ∧ (S = 24.5) :=
by
  sorry

end NUMINAMATH_GPT_sachin_age_l669_66946


namespace NUMINAMATH_GPT_pens_distributed_evenly_l669_66964

theorem pens_distributed_evenly (S : ℕ) (P : ℕ) (pencils : ℕ) 
  (hS : S = 10) (hpencils : pencils = 920) 
  (h_pencils_distributed : pencils % S = 0) 
  (h_pens_distributed : P % S = 0) : 
  ∃ k : ℕ, P = 10 * k :=
by 
  sorry

end NUMINAMATH_GPT_pens_distributed_evenly_l669_66964


namespace NUMINAMATH_GPT_find_n_modulo_conditions_l669_66922

theorem find_n_modulo_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n % 7 = -3137 % 7 ∧ (n = 1 ∨ n = 8) := sorry

end NUMINAMATH_GPT_find_n_modulo_conditions_l669_66922


namespace NUMINAMATH_GPT_milk_production_l669_66982

variables (a b c d e : ℕ) (h1 : a > 0) (h2 : c > 0)

def summer_rate := b / (a * c) -- Rate in summer per cow per day
def winter_rate := 2 * summer_rate -- Rate in winter per cow per day

noncomputable def total_milk_produced := (d * summer_rate * e) + (d * winter_rate * e)

theorem milk_production (h : d > 0) : total_milk_produced a b c d e = 3 * b * d * e / (a * c) :=
by sorry

end NUMINAMATH_GPT_milk_production_l669_66982


namespace NUMINAMATH_GPT_rectangular_field_area_l669_66902

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l669_66902


namespace NUMINAMATH_GPT_speed_of_stream_l669_66916

variable (b s : ℝ)

-- Define the conditions from the problem
def downstream_condition := (100 : ℝ) / 4 = b + s
def upstream_condition := (75 : ℝ) / 15 = b - s

theorem speed_of_stream (h1 : downstream_condition b s) (h2: upstream_condition b s) : s = 10 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_stream_l669_66916


namespace NUMINAMATH_GPT_students_joined_l669_66911

theorem students_joined (A X : ℕ) (h1 : 100 * A = 5000) (h2 : (100 + X) * (A - 10) = 5400) :
  X = 35 :=
by
  sorry

end NUMINAMATH_GPT_students_joined_l669_66911


namespace NUMINAMATH_GPT_units_digit_of_7_power_19_l669_66903

theorem units_digit_of_7_power_19 : (7^19) % 10 = 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_7_power_19_l669_66903


namespace NUMINAMATH_GPT_ratio_wealth_citizen_XY_l669_66989

noncomputable def wealth_ratio_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : ℝ :=
  let pop_X := 0.4 * P
  let wealth_X_before_tax := 0.5 * W
  let tax_X := 0.1 * wealth_X_before_tax
  let wealth_X_after_tax := wealth_X_before_tax - tax_X
  let wealth_per_citizen_X := wealth_X_after_tax / pop_X

  let pop_Y := 0.3 * P
  let wealth_Y := 0.6 * W
  let wealth_per_citizen_Y := wealth_Y / pop_Y

  wealth_per_citizen_X / wealth_per_citizen_Y

theorem ratio_wealth_citizen_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : 
  wealth_ratio_XY P W h1 h2 = 9 / 16 := 
by
  sorry

end NUMINAMATH_GPT_ratio_wealth_citizen_XY_l669_66989


namespace NUMINAMATH_GPT_coordinates_on_y_axis_l669_66998

theorem coordinates_on_y_axis (m : ℝ) (h : m + 1 = 0) : (m + 1, m + 4) = (0, 3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_on_y_axis_l669_66998


namespace NUMINAMATH_GPT_find_rate_percent_l669_66913

-- Definitions
def simpleInterest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Given conditions
def principal : ℕ := 900
def time : ℕ := 4
def simpleInterestValue : ℕ := 160

-- Rate percent
theorem find_rate_percent : 
  ∃ R : ℕ, simpleInterest principal R time = simpleInterestValue :=
by
  sorry

end NUMINAMATH_GPT_find_rate_percent_l669_66913


namespace NUMINAMATH_GPT_sum_powers_divisible_by_13_l669_66905

-- Statement of the problem in Lean
theorem sum_powers_divisible_by_13 (a b p : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : p = 13) :
  (a^1974 + b^1974) % p = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_powers_divisible_by_13_l669_66905


namespace NUMINAMATH_GPT_number_of_balls_l669_66995

theorem number_of_balls (x : ℕ) (h : x - 20 = 30 - x) : x = 25 :=
sorry

end NUMINAMATH_GPT_number_of_balls_l669_66995


namespace NUMINAMATH_GPT_hadley_total_walking_distance_l669_66975

-- Definitions of the distances walked to each location
def distance_grocery_store : ℕ := 2
def distance_pet_store : ℕ := distance_grocery_store - 1
def distance_home : ℕ := 4 - 1

-- Total distance walked by Hadley
def total_distance : ℕ := distance_grocery_store + distance_pet_store + distance_home

-- Statement to be proved
theorem hadley_total_walking_distance : total_distance = 6 := by
  sorry

end NUMINAMATH_GPT_hadley_total_walking_distance_l669_66975


namespace NUMINAMATH_GPT_ratio_of_capitals_l669_66917

-- Variables for the capitals of Ashok and Pyarelal
variables (A P : ℕ)

-- Given conditions
def total_loss := 670
def pyarelal_loss := 603
def ashok_loss := total_loss - pyarelal_loss

-- Proof statement: the ratio of Ashok's capital to Pyarelal's capital
theorem ratio_of_capitals : ashok_loss * P = total_loss * pyarelal_loss - pyarelal_loss * P → A * pyarelal_loss = P * ashok_loss :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_capitals_l669_66917


namespace NUMINAMATH_GPT_least_width_l669_66988

theorem least_width (w : ℝ) (h_nonneg : w ≥ 0) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end NUMINAMATH_GPT_least_width_l669_66988


namespace NUMINAMATH_GPT_conic_sections_union_l669_66971

theorem conic_sections_union :
  ∀ (x y : ℝ), (y^4 - 4*x^4 = 2*y^2 - 1) ↔ 
               (y^2 - 2*x^2 = 1) ∨ (y^2 + 2*x^2 = 1) := 
by
  sorry

end NUMINAMATH_GPT_conic_sections_union_l669_66971


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l669_66932

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : ∀ x, -3 < x ∧ x < 1/2 ↔ cx^2 + bx + a < 0) :
  ∀ x, -1/3 ≤ x ∧ x ≤ 2 ↔ ax^2 + bx + c ≥ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l669_66932


namespace NUMINAMATH_GPT_chord_length_l669_66986

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l669_66986


namespace NUMINAMATH_GPT_determine_triangle_area_l669_66980

noncomputable def triangle_area_proof : Prop :=
  let height : ℝ := 2
  let angle_ratio : ℝ := 2 / 1
  let smaller_base_part : ℝ := 1
  let larger_base_part : ℝ := 7 / 3
  let base := smaller_base_part + larger_base_part
  let area := (1 / 2) * base * height
  area = 11 / 3

theorem determine_triangle_area : triangle_area_proof :=
by
  sorry

end NUMINAMATH_GPT_determine_triangle_area_l669_66980


namespace NUMINAMATH_GPT_gcd_47_power5_1_l669_66956
-- Import the necessary Lean library

-- Mathematically equivalent proof problem in Lean 4
theorem gcd_47_power5_1 (a b : ℕ) (h1 : a = 47^5 + 1) (h2 : b = 47^5 + 47^3 + 1) :
  Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_47_power5_1_l669_66956


namespace NUMINAMATH_GPT_option_C_is_always_odd_l669_66935

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_C_is_always_odd (k : ℤ) : is_odd (2007 + 2 * k ^ 2) :=
sorry

end NUMINAMATH_GPT_option_C_is_always_odd_l669_66935


namespace NUMINAMATH_GPT_linear_function_quadrants_l669_66947

theorem linear_function_quadrants
  (k : ℝ) (h₀ : k ≠ 0) (h₁ : ∀ x : ℝ, x > 0 → k*x < 0) :
  (∃ x > 0, 2*x + k > 0) ∧
  (∃ x > 0, 2*x + k < 0) ∧
  (∃ x < 0, 2*x + k < 0) :=
  by
  sorry

end NUMINAMATH_GPT_linear_function_quadrants_l669_66947


namespace NUMINAMATH_GPT_triangle_problem_l669_66983

theorem triangle_problem (n : ℕ) (h : 1 < n ∧ n < 4) : n = 2 ∨ n = 3 :=
by
  -- Valid realizability proof omitted
  sorry

end NUMINAMATH_GPT_triangle_problem_l669_66983


namespace NUMINAMATH_GPT_ratio_of_games_played_to_losses_l669_66967

-- Definitions based on the conditions
def total_games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := total_games_played - games_won

-- The proof problem
theorem ratio_of_games_played_to_losses : (total_games_played / Nat.gcd total_games_played games_lost) = 2 ∧ (games_lost / Nat.gcd total_games_played games_lost) = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_games_played_to_losses_l669_66967


namespace NUMINAMATH_GPT_line_equation_parallel_l669_66961

theorem line_equation_parallel (x₁ y₁ m : ℝ) (h₁ : (x₁, y₁) = (1, -2)) (h₂ : m = 2) :
  ∃ a b c : ℝ, a * x₁ + b * y₁ + c = 0 ∧ a * 2 + b * 1 + c = 4 := by
sorry

end NUMINAMATH_GPT_line_equation_parallel_l669_66961


namespace NUMINAMATH_GPT_correct_option_l669_66959

-- Definitions for universe set, and subsets A and B
def S : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- The proof goal
theorem correct_option : A ⊆ S \ B :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l669_66959


namespace NUMINAMATH_GPT_greatest_multiple_l669_66963

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end NUMINAMATH_GPT_greatest_multiple_l669_66963


namespace NUMINAMATH_GPT_little_johns_money_left_l669_66990

def J_initial : ℝ := 7.10
def S : ℝ := 1.05
def F : ℝ := 1.00

theorem little_johns_money_left :
  J_initial - (S + 2 * F) = 4.05 :=
by sorry

end NUMINAMATH_GPT_little_johns_money_left_l669_66990


namespace NUMINAMATH_GPT_pen_ratio_l669_66929

theorem pen_ratio 
  (Dorothy_pens Julia_pens Robert_pens : ℕ)
  (pen_cost total_cost : ℚ)
  (h1 : Dorothy_pens = Julia_pens / 2)
  (h2 : Robert_pens = 4)
  (h3 : pen_cost = 1.5)
  (h4 : total_cost = 33)
  (h5 : total_cost / pen_cost = Dorothy_pens + Julia_pens + Robert_pens) :
  (Julia_pens / Robert_pens : ℚ) = 3 :=
  sorry

end NUMINAMATH_GPT_pen_ratio_l669_66929


namespace NUMINAMATH_GPT_total_shoes_l669_66957

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end NUMINAMATH_GPT_total_shoes_l669_66957
