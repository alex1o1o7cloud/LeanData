import Mathlib

namespace NUMINAMATH_CALUDE_wire_length_for_cube_l3821_382113

-- Define the length of one edge of the cube
def edge_length : ℝ := 13

-- Define the number of edges in a cube
def cube_edges : ℕ := 12

-- Theorem stating the total wire length needed for the cube
theorem wire_length_for_cube : edge_length * cube_edges = 156 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_for_cube_l3821_382113


namespace NUMINAMATH_CALUDE_tan_alpha_equals_negative_one_l3821_382111

theorem tan_alpha_equals_negative_one (α : Real) 
  (h1 : |Real.sin α| = |Real.cos α|) 
  (h2 : α > Real.pi / 2 ∧ α < Real.pi) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_negative_one_l3821_382111


namespace NUMINAMATH_CALUDE_water_added_to_container_l3821_382101

/-- The amount of water added to a container -/
def water_added (capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) : ℝ :=
  capacity * final_fraction - capacity * initial_fraction

/-- Theorem stating the amount of water added to the container -/
theorem water_added_to_container : 
  water_added 80 0.4 0.75 = 28 := by sorry

end NUMINAMATH_CALUDE_water_added_to_container_l3821_382101


namespace NUMINAMATH_CALUDE_chess_tournament_ties_l3821_382117

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  num_players : Nat
  points_win : Rat
  points_loss : Rat
  points_tie : Rat
  total_games : Nat
  total_points : Rat
  best_three_points : Rat
  last_nine_points : Rat

/-- The main theorem to be proved -/
theorem chess_tournament_ties (t : ChessTournament) : 
  t.num_players = 14 ∧ 
  t.points_win = 1 ∧ 
  t.points_loss = 0 ∧ 
  t.points_tie = 1/2 ∧ 
  t.total_games = 91 ∧ 
  t.total_points = 91 ∧
  t.best_three_points = t.last_nine_points ∧
  t.best_three_points = 36 →
  ∃ (num_ties : Nat), num_ties = 29 ∧ 
    (∀ (other_num_ties : Nat), other_num_ties > num_ties → 
      ¬(∃ (valid_tournament : ChessTournament), 
        valid_tournament.num_players = 14 ∧
        valid_tournament.points_win = 1 ∧
        valid_tournament.points_loss = 0 ∧
        valid_tournament.points_tie = 1/2 ∧
        valid_tournament.total_games = 91 ∧
        valid_tournament.total_points = 91 ∧
        valid_tournament.best_three_points = valid_tournament.last_nine_points ∧
        valid_tournament.best_three_points = 36)) :=
by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_ties_l3821_382117


namespace NUMINAMATH_CALUDE_multiplication_exponent_rule_l3821_382134

theorem multiplication_exponent_rule (a : ℝ) (h : a ≠ 0) : a * a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_exponent_rule_l3821_382134


namespace NUMINAMATH_CALUDE_ellipse_area_lower_bound_l3821_382180

/-- Given a right-angled triangle with area t, where the endpoints of its hypotenuse
    lie at the foci of an ellipse and the third vertex lies on the ellipse,
    the area of the ellipse is at least √2πt. -/
theorem ellipse_area_lower_bound (t : ℝ) (a b c : ℝ) (h1 : 0 < t) (h2 : 0 < b) (h3 : b < a)
    (h4 : a^2 = b^2 + c^2) (h5 : t = b^2) : π * a * b ≥ Real.sqrt 2 * π * t :=
by sorry

end NUMINAMATH_CALUDE_ellipse_area_lower_bound_l3821_382180


namespace NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l3821_382163

-- Define a function to convert octal to decimal
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

-- Theorem statement
theorem octal_123_equals_decimal_83 :
  octal_to_decimal [3, 2, 1] = 83 := by
  sorry

end NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l3821_382163


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l3821_382126

/-- Proves that the ratio of a rectangular field's length to its width is 2:1,
    given specific conditions about the field and a pond within it. -/
theorem field_length_width_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 48 →
  pond_side = 8 →
  pond_side * pond_side = (field_length * field_width) / 18 →
  field_length / field_width = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l3821_382126


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3821_382138

theorem completing_square_equivalence :
  ∀ x : ℝ, 2 * x^2 - 4 * x - 7 = 0 ↔ (x - 1)^2 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3821_382138


namespace NUMINAMATH_CALUDE_window_treatment_cost_l3821_382166

/-- The number of windows Laura needs to buy window treatments for -/
def num_windows : ℕ := 3

/-- The cost of sheers for one window in cents -/
def sheer_cost : ℕ := 4000

/-- The cost of drapes for one window in cents -/
def drape_cost : ℕ := 6000

/-- The total cost for all windows in cents -/
def total_cost : ℕ := 30000

/-- Theorem stating that the number of windows is correct given the costs -/
theorem window_treatment_cost : 
  (sheer_cost + drape_cost) * num_windows = total_cost := by
  sorry


end NUMINAMATH_CALUDE_window_treatment_cost_l3821_382166


namespace NUMINAMATH_CALUDE_steve_gum_pieces_l3821_382133

theorem steve_gum_pieces (initial_gum : ℕ) (total_gum : ℕ) (h1 : initial_gum = 38) (h2 : total_gum = 54) :
  total_gum - initial_gum = 16 := by
  sorry

end NUMINAMATH_CALUDE_steve_gum_pieces_l3821_382133


namespace NUMINAMATH_CALUDE_frank_candy_purchase_l3821_382164

/-- The number of candies Frank can buy with his arcade tickets -/
def candies_bought (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost

/-- Theorem: Frank can buy 7 candies with his arcade tickets -/
theorem frank_candy_purchase :
  candies_bought 33 9 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_purchase_l3821_382164


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l3821_382175

theorem greatest_three_digit_multiple_of_23 : ∀ n : ℕ, n < 1000 → n ≥ 100 → n % 23 = 0 → n ≤ 989 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l3821_382175


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3821_382130

theorem cube_equation_solution (c : ℤ) : 
  c^3 + 3*c + 3/c + 1/c^3 = 8 → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3821_382130


namespace NUMINAMATH_CALUDE_billy_weight_l3821_382176

/-- Given the weights of Billy, Brad, and Carl, prove that Billy weighs 159 pounds. -/
theorem billy_weight (billy brad carl : ℕ) 
  (h1 : billy = brad + 9)
  (h2 : brad = carl + 5)
  (h3 : carl = 145) : 
  billy = 159 := by
  sorry

end NUMINAMATH_CALUDE_billy_weight_l3821_382176


namespace NUMINAMATH_CALUDE_inequality_equivalence_system_of_inequalities_equivalence_l3821_382189

theorem inequality_equivalence (x : ℝ) :
  (1 - (x - 3) / 6 > x / 3) ↔ (x < 3) :=
sorry

theorem system_of_inequalities_equivalence (x : ℝ) :
  (x + 1 ≥ 3 * (x - 3) ∧ (x + 2) / 3 - (x - 1) / 4 > 1) ↔ (1 < x ∧ x ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_system_of_inequalities_equivalence_l3821_382189


namespace NUMINAMATH_CALUDE_circle_equation_line_equation_l3821_382152

/-- A circle C passing through (2,-1), tangent to x+y=1, with center on y=-2x -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : center.1^2 + (center.2 + 1)^2 = radius^2
  tangent_to_line : |center.1 + center.2 - 1| / Real.sqrt 2 = radius
  center_on_line : center.2 = -2 * center.1

/-- A line passing through the origin and cutting a chord of length 2 on CircleC -/
structure LineL (c : CircleC) where
  slope : ℝ
  passes_origin : True
  cuts_chord : (2 * c.radius / Real.sqrt (1 + slope^2))^2 = 4

theorem circle_equation (c : CircleC) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

theorem line_equation (c : CircleC) (l : LineL c) :
  (l.slope = 0 ∧ ∀ x y : ℝ, y = l.slope * x ↔ x = 0) ∨
  (l.slope = -3/4 ∧ ∀ x y : ℝ, y = l.slope * x ↔ y = -3/4 * x) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_line_equation_l3821_382152


namespace NUMINAMATH_CALUDE_lucy_cake_packs_l3821_382174

/-- Represents the number of packs of cookies Lucy bought -/
def cookie_packs : ℕ := 23

/-- Represents the total number of grocery packs Lucy bought -/
def total_packs : ℕ := 27

/-- Represents the number of cake packs Lucy bought -/
def cake_packs : ℕ := total_packs - cookie_packs

/-- Proves that the number of cake packs Lucy bought is equal to 4 -/
theorem lucy_cake_packs : cake_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucy_cake_packs_l3821_382174


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l3821_382187

/-- The area of the union of a rectangle and a circle -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 8
  let rectangle_height : ℝ := 12
  let circle_radius : ℝ := 8
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 48 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l3821_382187


namespace NUMINAMATH_CALUDE_sum_of_differences_l3821_382123

def S : Finset ℕ := Finset.range 9

def diff_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if i > j then 2^i - 2^j else 0))

theorem sum_of_differences : diff_sum S = 3096 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_differences_l3821_382123


namespace NUMINAMATH_CALUDE_sunflower_plants_count_l3821_382183

/-- The number of corn plants -/
def corn_plants : ℕ := 81

/-- The number of tomato plants -/
def tomato_plants : ℕ := 63

/-- The maximum number of plants in one row -/
def max_plants_per_row : ℕ := 9

/-- The number of rows for corn plants -/
def corn_rows : ℕ := corn_plants / max_plants_per_row

/-- The number of rows for tomato plants -/
def tomato_rows : ℕ := tomato_plants / max_plants_per_row

/-- The number of rows for sunflower plants -/
def sunflower_rows : ℕ := max corn_rows tomato_rows

/-- The theorem stating the number of sunflower plants -/
theorem sunflower_plants_count : 
  ∃ (sunflower_plants : ℕ), 
    sunflower_plants = sunflower_rows * max_plants_per_row ∧ 
    sunflower_plants = 81 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_plants_count_l3821_382183


namespace NUMINAMATH_CALUDE_renovation_problem_l3821_382120

/-- Proves that given the conditions of the renovation problem, the number of days worked is 7 -/
theorem renovation_problem (hours_per_day : ℕ) (total_cost : ℕ) (rate_pro1 : ℕ) :
  hours_per_day = 6 →
  total_cost = 1260 →
  rate_pro1 = 15 →
  ∃ (days : ℕ), days * hours_per_day * (rate_pro1 + rate_pro1) = total_cost ∧ days = 7 :=
by sorry

end NUMINAMATH_CALUDE_renovation_problem_l3821_382120


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_three_l3821_382142

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d ≥ 0 ∧ d < 1 then d / (1 - (10 * d - ⌊10 * d⌋)) else d

theorem reciprocal_of_repeating_three : 
  (repeating_decimal_to_fraction (1/3 : ℚ))⁻¹ = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_three_l3821_382142


namespace NUMINAMATH_CALUDE_product_53_57_l3821_382190

theorem product_53_57 (h : 2021 = 43 * 47) : 53 * 57 = 3021 := by
  sorry

end NUMINAMATH_CALUDE_product_53_57_l3821_382190


namespace NUMINAMATH_CALUDE_total_appetizers_l3821_382107

def hotdogs : ℕ := 60
def cheese_pops : ℕ := 40
def chicken_nuggets : ℕ := 80
def mini_quiches : ℕ := 100
def stuffed_mushrooms : ℕ := 50

theorem total_appetizers : 
  hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms = 330 := by
  sorry

end NUMINAMATH_CALUDE_total_appetizers_l3821_382107


namespace NUMINAMATH_CALUDE_equation_solutions_l3821_382181

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 9 = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, (-x)^3 = (-8)^2 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3821_382181


namespace NUMINAMATH_CALUDE_least_questions_for_probability_l3821_382171

theorem least_questions_for_probability (n : ℕ) : n ≥ 4 ↔ (1/2 : ℝ)^n < 1/10 := by sorry

end NUMINAMATH_CALUDE_least_questions_for_probability_l3821_382171


namespace NUMINAMATH_CALUDE_max_pairs_sum_l3821_382158

theorem max_pairs_sum (k : ℕ) (a b : ℕ → ℕ) : 
  (∀ i : ℕ, i < k → a i < b i) →
  (∀ i j : ℕ, i < k → j < k → i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) →
  (∀ i : ℕ, i < k → a i ∈ Finset.range 4019 ∧ b i ∈ Finset.range 4019) →
  (∀ i : ℕ, i < k → a i + b i ≤ 4019) →
  (∀ i j : ℕ, i < k → j < k → i ≠ j → a i + b i ≠ a j + b j) →
  k ≤ 1607 :=
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l3821_382158


namespace NUMINAMATH_CALUDE_f_properties_l3821_382182

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos φ + 2 * Real.cos (ω * x) * Real.sin φ

theorem f_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ∃ φ',
    (∀ x, f ω φ x = 2 * Real.sin (2 * x + φ')) ∧
    (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≤ 2) ∧
    (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≥ 0) ∧
    (∃ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x = 2) ∧
    ((∃ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x = 0) ∨
     (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≥ 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3821_382182


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3821_382153

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  1 / (x + 1) + 4 / (y + 2) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3821_382153


namespace NUMINAMATH_CALUDE_product_probability_l3821_382195

def paco_range : Finset ℕ := Finset.range 5
def manu_range : Finset ℕ := Finset.range 15

def is_product_less_than_50 (p m : ℕ) : Bool :=
  (p + 1) * (m + 1) < 50

def favorable_outcomes : Finset (ℕ × ℕ) :=
  paco_range.product manu_range |>.filter (λ (p, m) ↦ is_product_less_than_50 p m)

def total_outcomes : ℕ := paco_range.card * manu_range.card

theorem product_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes = 22 / 25 := by sorry

end NUMINAMATH_CALUDE_product_probability_l3821_382195


namespace NUMINAMATH_CALUDE_first_class_students_l3821_382135

theorem first_class_students (avg_first : ℝ) (students_second : ℕ) (avg_second : ℝ) (avg_all : ℝ)
  (h1 : avg_first = 30)
  (h2 : students_second = 50)
  (h3 : avg_second = 60)
  (h4 : avg_all = 48.75) :
  ∃ students_first : ℕ,
    students_first * avg_first + students_second * avg_second =
    (students_first + students_second) * avg_all ∧
    students_first = 30 :=
by sorry

end NUMINAMATH_CALUDE_first_class_students_l3821_382135


namespace NUMINAMATH_CALUDE_ada_original_seat_l3821_382185

-- Define the seats
inductive Seat : Type
| one : Seat
| two : Seat
| three : Seat
| four : Seat
| five : Seat
| six : Seat

-- Define the friends
inductive Friend : Type
| ada : Friend
| bea : Friend
| ceci : Friend
| dee : Friend
| edie : Friend
| fred : Friend

-- Define the seating arrangement as a function from Friend to Seat
def Seating := Friend → Seat

-- Define the movement function
def move (s : Seating) : Seating :=
  fun f => match f with
    | Friend.bea => match s Friend.bea with
      | Seat.one => Seat.two
      | Seat.two => Seat.three
      | Seat.three => Seat.four
      | Seat.four => Seat.five
      | Seat.five => Seat.six
      | Seat.six => Seat.six
    | Friend.ceci => match s Friend.ceci with
      | Seat.one => Seat.one
      | Seat.two => Seat.one
      | Seat.three => Seat.one
      | Seat.four => Seat.two
      | Seat.five => Seat.three
      | Seat.six => Seat.four
    | Friend.dee => s Friend.edie
    | Friend.edie => s Friend.dee
    | Friend.fred => s Friend.fred
    | Friend.ada => s Friend.ada

-- Theorem stating Ada's original seat
theorem ada_original_seat (initial : Seating) :
  (move initial) Friend.ada = Seat.one →
  initial Friend.ada = Seat.two :=
by
  sorry


end NUMINAMATH_CALUDE_ada_original_seat_l3821_382185


namespace NUMINAMATH_CALUDE_age_of_replaced_man_is_44_l3821_382154

/-- The age of the other replaced man given the conditions of the problem -/
def age_of_replaced_man (initial_men_count : ℕ) (age_increase : ℕ) (known_man_age : ℕ) (women_avg_age : ℕ) : ℕ :=
  44

/-- Theorem stating that the age of the other replaced man is 44 years old -/
theorem age_of_replaced_man_is_44 
  (initial_men_count : ℕ) 
  (age_increase : ℕ) 
  (known_man_age : ℕ) 
  (women_avg_age : ℕ) 
  (h1 : initial_men_count = 6)
  (h2 : age_increase = 3)
  (h3 : known_man_age = 24)
  (h4 : women_avg_age = 34) :
  age_of_replaced_man initial_men_count age_increase known_man_age women_avg_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_age_of_replaced_man_is_44_l3821_382154


namespace NUMINAMATH_CALUDE_power_three_sum_l3821_382159

theorem power_three_sum (m n : ℕ+) (x y : ℝ) 
  (hx : 3^(m.val) = x) 
  (hy : 9^(n.val) = y) : 
  3^(m.val + 2*n.val) = x * y := by
sorry

end NUMINAMATH_CALUDE_power_three_sum_l3821_382159


namespace NUMINAMATH_CALUDE_ellipse_foci_product_l3821_382122

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 / 16 + P.2^2 / 12 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the dot product condition
def satisfies_dot_product (P : ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 9

-- Theorem statement
theorem ellipse_foci_product (P : ℝ × ℝ) :
  is_on_ellipse P → satisfies_dot_product P →
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_product_l3821_382122


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l3821_382115

theorem initial_number_of_persons (n : ℕ) 
  (h1 : (3 : ℝ) * n = 24) : n = 8 := by
  sorry

#check initial_number_of_persons

end NUMINAMATH_CALUDE_initial_number_of_persons_l3821_382115


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l3821_382156

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 8
def cooking_time_per_potato : ℕ := 9

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l3821_382156


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l3821_382104

def P (x : ℝ) : Prop := |x - 2| ≤ 3

def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬(P x)) := by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l3821_382104


namespace NUMINAMATH_CALUDE_reflection_sum_l3821_382103

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line of reflection y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if two points are reflections of each other across a given line -/
def areReflections (A B : Point) (L : Line) : Prop :=
  let midpoint : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
  (midpoint.y = L.m * midpoint.x + L.b) ∧
  (L.m = -(B.x - A.x) / (B.y - A.y))

/-- The main theorem -/
theorem reflection_sum (A B : Point) (L : Line) :
  A = ⟨2, 3⟩ → B = ⟨10, 7⟩ → areReflections A B L → L.m + L.b = 15 := by
  sorry


end NUMINAMATH_CALUDE_reflection_sum_l3821_382103


namespace NUMINAMATH_CALUDE_circle_ranges_l3821_382100

/-- The equation of a circle with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2)*y + 16*m^4 + 9 = 0

/-- The range of m for which the equation represents a circle -/
def m_range (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

/-- The range of the radius r of the circle -/
def r_range (r : ℝ) : Prop :=
  0 < r ∧ r ≤ 4/Real.sqrt 7

/-- Theorem stating the ranges of m and r for the given circle equation -/
theorem circle_ranges :
  (∃ x y : ℝ, circle_equation x y m) → m_range m ∧ (∃ r : ℝ, r_range r) :=
by sorry

end NUMINAMATH_CALUDE_circle_ranges_l3821_382100


namespace NUMINAMATH_CALUDE_inequality_condition_l3821_382114

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 5| + |x - 2| < b) ↔ b > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l3821_382114


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3821_382196

theorem probability_of_white_ball 
  (prob_red : ℝ) 
  (prob_black : ℝ) 
  (h1 : prob_red = 0.4) 
  (h2 : prob_black = 0.25) 
  (h3 : prob_red + prob_black + (1 - prob_red - prob_black) = 1) :
  1 - prob_red - prob_black = 0.35 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3821_382196


namespace NUMINAMATH_CALUDE_age_difference_brother_cousin_l3821_382150

/-- Proves that the age difference between Lexie's brother and cousin is 5 years -/
theorem age_difference_brother_cousin : 
  ∀ (lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ),
  lexie_age = 8 →
  grandma_age = 68 →
  lexie_age = brother_age + 6 →
  sister_age = 2 * lexie_age →
  uncle_age + 12 = grandma_age →
  cousin_age = brother_age + 5 →
  cousin_age - brother_age = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_brother_cousin_l3821_382150


namespace NUMINAMATH_CALUDE_valid_triangle_divisions_l3821_382173

/-- Represents a division of an equilateral triangle into smaller triangles -/
structure TriangleDivision where
  n : ℕ  -- number of smaller triangles
  k : ℕ  -- number of identical polygons

/-- Predicate to check if a division is valid -/
def is_valid_division (d : TriangleDivision) : Prop :=
  d.n = 36 ∧ d.k ∣ d.n ∧ 
  (d.k = 1 ∨ d.k = 3 ∨ d.k = 4 ∨ d.k = 9 ∨ d.k = 12 ∨ d.k = 36)

/-- Theorem stating the valid divisions of the triangle -/
theorem valid_triangle_divisions :
  ∀ d : TriangleDivision, is_valid_division d ↔ 
    (d.k = 1 ∨ d.k = 3 ∨ d.k = 4 ∨ d.k = 9 ∨ d.k = 12 ∨ d.k = 36) :=
by sorry

end NUMINAMATH_CALUDE_valid_triangle_divisions_l3821_382173


namespace NUMINAMATH_CALUDE_pet_store_parrots_l3821_382193

/-- The number of bird cages in the pet store -/
def num_cages : ℝ := 6.0

/-- The number of parakeets in the pet store -/
def num_parakeets : ℝ := 2.0

/-- The average number of birds that can occupy one cage -/
def birds_per_cage : ℝ := 1.333333333

/-- The number of parrots in the pet store -/
def num_parrots : ℝ := 6.0

/-- Theorem stating that the number of parrots in the pet store is 6.0 -/
theorem pet_store_parrots : 
  num_parrots = num_cages * birds_per_cage - num_parakeets := by
  sorry

end NUMINAMATH_CALUDE_pet_store_parrots_l3821_382193


namespace NUMINAMATH_CALUDE_three_integers_ratio_l3821_382155

theorem three_integers_ratio : ∀ (a b c : ℤ),
  (a : ℚ) / b = 2 / 5 ∧ 
  (b : ℚ) / c = 5 / 8 ∧ 
  ((a + 6 : ℚ) / b = 1 / 3) →
  a = 36 ∧ b = 90 ∧ c = 144 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_ratio_l3821_382155


namespace NUMINAMATH_CALUDE_range_of_m_l3821_382147

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧ 
   (∃ y : ℝ, (y - m + 1) * (y - m - 1) ≤ 0 ∧ |y - 3| > 2)) →
  2 ≤ m ∧ m ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3821_382147


namespace NUMINAMATH_CALUDE_certain_number_proof_l3821_382125

theorem certain_number_proof (x : ℝ) : x + 6 = 8 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3821_382125


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3821_382129

theorem quadratic_root_property (m : ℝ) : 
  m^2 - 3*m + 1 = 0 → 2*m^2 - 6*m - 2024 = -2026 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3821_382129


namespace NUMINAMATH_CALUDE_tan_negative_23pi_over_6_sin_75_degrees_l3821_382106

-- Part 1
theorem tan_negative_23pi_over_6 : 
  Real.tan (-23 * π / 6) = Real.sqrt 3 / 3 := by sorry

-- Part 2
theorem sin_75_degrees : 
  Real.sin (75 * π / 180) = (Real.sqrt 2 + Real.sqrt 6) / 4 := by sorry

end NUMINAMATH_CALUDE_tan_negative_23pi_over_6_sin_75_degrees_l3821_382106


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_solution_l3821_382110

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y ≥ 22) ↔ (y ≤ -5) := by sorry

theorem smallest_solution : ∃ (y : ℤ), (7 - 3 * y ≥ 22) ∧ ∀ (z : ℤ), (7 - 3 * z ≥ 22) → (y ≤ z) := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_solution_l3821_382110


namespace NUMINAMATH_CALUDE_basic_computer_price_l3821_382186

/-- The price of the basic computer and printer total $2,500, and the printer costs 1/6 of the total when paired with an enhanced computer $500 more expensive than the basic one. -/
theorem basic_computer_price (basic_price printer_price : ℝ) 
  (h1 : basic_price + printer_price = 2500)
  (h2 : printer_price = (1 / 6) * ((basic_price + 500) + printer_price)) :
  basic_price = 2000 := by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l3821_382186


namespace NUMINAMATH_CALUDE_max_distance_to_ellipse_l3821_382118

/-- The maximum distance between a fixed point A(0,2) and any point P on the ellipse x²/4 + y² = 1 is 2√21/3 -/
theorem max_distance_to_ellipse :
  let A : ℝ × ℝ := (0, 2)
  let ellipse := {P : ℝ × ℝ | P.1^2/4 + P.2^2 = 1}
  ∃ (M : ℝ), M = 2 * Real.sqrt 21 / 3 ∧
    ∀ P ∈ ellipse, Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) ≤ M ∧
    ∃ P ∈ ellipse, Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = M :=
by
  sorry


end NUMINAMATH_CALUDE_max_distance_to_ellipse_l3821_382118


namespace NUMINAMATH_CALUDE_real_roots_condition_specific_roots_condition_l3821_382179

variable (m : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the quadratic equation
def quadratic (x : ℝ) := x^2 - 6*x + (4*m + 1)

-- Theorem 1: For real roots, m ≤ 2
theorem real_roots_condition : (∃ x : ℝ, quadratic m x = 0) → m ≤ 2 := by sorry

-- Theorem 2: If x₁ and x₂ are roots and x₁² + x₂² = 26, then m = 1
theorem specific_roots_condition : 
  quadratic m x₁ = 0 → quadratic m x₂ = 0 → x₁^2 + x₂^2 = 26 → m = 1 := by sorry

end NUMINAMATH_CALUDE_real_roots_condition_specific_roots_condition_l3821_382179


namespace NUMINAMATH_CALUDE_solution_set_equals_given_solutions_l3821_382131

/-- The set of solutions to the system of equations -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {xyz : ℝ × ℝ × ℝ | let (x, y, z) := xyz
                     x + y * z = 20 ∧
                     y + z * x = 20 ∧
                     z + x * y = 20}

/-- The given set of solutions -/
def GivenSolutions : Set (ℝ × ℝ × ℝ) :=
  {(4, 4, 4), (-5, -5, -5), (1, 1, 19), (19, 1, 1), (1, 19, 1)}

/-- Theorem stating that the solution set equals the given solutions -/
theorem solution_set_equals_given_solutions : SolutionSet = GivenSolutions := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equals_given_solutions_l3821_382131


namespace NUMINAMATH_CALUDE_problem_solution_l3821_382199

def sum_of_integers (a b : ℕ) : ℕ :=
  ((b - a + 1) * (a + b)) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem problem_solution :
  let x := sum_of_integers 40 60
  let y := count_even_integers 40 60
  x + y = 1061 → y = 11 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3821_382199


namespace NUMINAMATH_CALUDE_negation_equivalence_l3821_382145

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨ (Odd a ∧ Even b ∧ Odd c) ∨ (Odd a ∧ Odd b ∧ Even c)

def all_odd_or_at_least_two_even (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∨ (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ all_odd_or_at_least_two_even a b c := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3821_382145


namespace NUMINAMATH_CALUDE_max_difference_bounded_l3821_382102

theorem max_difference_bounded (a : Fin 2017 → ℝ) 
  (h1 : a 1 = a 2017)
  (h2 : ∀ i : Fin 2015, |a i + a (i + 2) - 2 * a (i + 1)| ≤ 1) :
  ∃ M : ℝ, M = 508032 ∧ 
  (∀ i j : Fin 2017, i < j → |a i - a j| ≤ M) ∧
  (∃ i j : Fin 2017, i < j ∧ |a i - a j| = M) := by
sorry

end NUMINAMATH_CALUDE_max_difference_bounded_l3821_382102


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3821_382184

theorem rectangle_area_problem (p q : ℝ) : 
  q = (2/5) * p →  -- point (p, q) is on the line y = 2/5 x
  p * q = 90 →     -- area of the rectangle is 90
  p = 15 :=        -- prove that p = 15
by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3821_382184


namespace NUMINAMATH_CALUDE_f_min_max_l3821_382124

-- Define the function
def f (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

-- State the theorem
theorem f_min_max :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x y ≥ -1/3) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x y ≤ 9/8) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ f x y = -1/3) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ f x y = 9/8) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l3821_382124


namespace NUMINAMATH_CALUDE_limit_sin_squared_minus_tan_squared_over_x_fourth_l3821_382128

open Real

theorem limit_sin_squared_minus_tan_squared_over_x_fourth : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((sin x)^2 - (tan x)^2) / x^4 + 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sin_squared_minus_tan_squared_over_x_fourth_l3821_382128


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3821_382121

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3821_382121


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l3821_382108

theorem propositions_p_and_q : 
  (∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l3821_382108


namespace NUMINAMATH_CALUDE_divisiblity_condition_l3821_382132

theorem divisiblity_condition (M : ℕ) : 
  0 < M ∧ M < 10 → (5 ∣ 1989^M + M^1889 ↔ M = 1 ∨ M = 4) :=
by sorry

end NUMINAMATH_CALUDE_divisiblity_condition_l3821_382132


namespace NUMINAMATH_CALUDE_marks_future_age_l3821_382162

def amy_age : ℕ := 15
def age_difference : ℕ := 7
def years_in_future : ℕ := 5

theorem marks_future_age :
  amy_age + age_difference + years_in_future = 27 := by
  sorry

end NUMINAMATH_CALUDE_marks_future_age_l3821_382162


namespace NUMINAMATH_CALUDE_distinctly_marked_fraction_l3821_382168

/-- Proves that the fraction of a 15 by 24 rectangular region that is distinctly marked is 1/6,
    given that one-third of the rectangle is shaded and half of the shaded area is distinctly marked. -/
theorem distinctly_marked_fraction (length width : ℕ) (shaded_fraction marked_fraction : ℚ) :
  length = 15 →
  width = 24 →
  shaded_fraction = 1/3 →
  marked_fraction = 1/2 →
  (shaded_fraction * marked_fraction : ℚ) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_distinctly_marked_fraction_l3821_382168


namespace NUMINAMATH_CALUDE_union_M_N_equals_M_l3821_382172

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}

-- State the theorem
theorem union_M_N_equals_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_M_l3821_382172


namespace NUMINAMATH_CALUDE_max_min_values_l3821_382167

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  |5 * x + y| + |5 * x - y| = 20

-- Define the expression to be maximized/minimized
def expr (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

-- Statement of the theorem
theorem max_min_values :
  (∃ x y : ℝ, constraint x y ∧ expr x y = 124) ∧
  (∃ x y : ℝ, constraint x y ∧ expr x y = 3) ∧
  (∀ x y : ℝ, constraint x y → 3 ≤ expr x y ∧ expr x y ≤ 124) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_l3821_382167


namespace NUMINAMATH_CALUDE_min_value_xy_l3821_382109

theorem min_value_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 5/x + 3/y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 5/a + 3/b = 2 → x * y ≤ a * b :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_l3821_382109


namespace NUMINAMATH_CALUDE_total_swordfish_catch_l3821_382127

/-- The number of swordfish Shelly catches per trip -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches per trip -/
def sam_catch : ℕ := shelly_catch - 1

/-- The number of fishing trips -/
def num_trips : ℕ := 5

/-- The total number of swordfish caught by Shelly and Sam -/
def total_catch : ℕ := (shelly_catch + sam_catch) * num_trips

theorem total_swordfish_catch : total_catch = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_swordfish_catch_l3821_382127


namespace NUMINAMATH_CALUDE_no_non_zero_solution_l3821_382144

theorem no_non_zero_solution (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_solution_l3821_382144


namespace NUMINAMATH_CALUDE_roots_can_change_l3821_382148

-- Define the concept of a root being lost
def root_lost (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ¬(Real.tan (f x) = Real.tan (g x))

-- Define the concept of an extraneous root appearing
def extraneous_root (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ g x ∧ Real.tan (f x) = Real.tan (g x)

-- Theorem stating that roots can be lost and extraneous roots can appear
theorem roots_can_change (f g : ℝ → ℝ) : 
  (root_lost f g) ∧ (extraneous_root f g) := by
  sorry


end NUMINAMATH_CALUDE_roots_can_change_l3821_382148


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3821_382198

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 7 * x^2 - 4 * x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3821_382198


namespace NUMINAMATH_CALUDE_special_functions_properties_l3821_382137

/-- Two functions satisfying a specific functional equation -/
class SpecialFunctions (f g : ℝ → ℝ) : Prop where
  eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y
  f_zero : f 0 = 0
  f_nonzero : ∃ x : ℝ, f x ≠ 0

/-- f is an odd function -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- g is an even function -/
def IsEvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

/-- Main theorem: f is odd and g is even -/
theorem special_functions_properties {f g : ℝ → ℝ} [SpecialFunctions f g] :
    IsOddFunction f ∧ IsEvenFunction g := by
  sorry

end NUMINAMATH_CALUDE_special_functions_properties_l3821_382137


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3821_382165

/-- Represents a hyperbola with focus on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_relation : a^2 + b^2 = c^2

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The equation of an asymptote of a hyperbola -/
def asymptoteEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : asymptoteEquation h x y ↔ y = 2 * x)
  (h_focus : h.c = Real.sqrt 5) :
  standardEquation h x y ↔ x^2 - y^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3821_382165


namespace NUMINAMATH_CALUDE_initial_men_count_l3821_382157

/-- Represents the amount of food consumed by one person in one day -/
def FoodPerPersonPerDay : ℝ := 1

/-- Calculates the total amount of food -/
def TotalFood (initialMen : ℕ) : ℝ := initialMen * 22 * FoodPerPersonPerDay

/-- Calculates the amount of food consumed in the first two days -/
def FoodConsumedInTwoDays (initialMen : ℕ) : ℝ := initialMen * 2 * FoodPerPersonPerDay

/-- Calculates the remaining food after two days -/
def RemainingFood (initialMen : ℕ) : ℝ := TotalFood initialMen - FoodConsumedInTwoDays initialMen

theorem initial_men_count (initialMen : ℕ) : 
  TotalFood initialMen = initialMen * 22 * FoodPerPersonPerDay ∧
  RemainingFood initialMen = (initialMen + 760) * 10 * FoodPerPersonPerDay →
  initialMen = 760 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l3821_382157


namespace NUMINAMATH_CALUDE_willie_cream_purchase_l3821_382191

/-- The amount of cream Willie needs to buy given the total required amount and the amount he already has. -/
def cream_to_buy (total_required : ℕ) (available : ℕ) : ℕ :=
  total_required - available

/-- Theorem stating that Willie needs to buy 151 lbs. of cream. -/
theorem willie_cream_purchase : cream_to_buy 300 149 = 151 := by
  sorry

end NUMINAMATH_CALUDE_willie_cream_purchase_l3821_382191


namespace NUMINAMATH_CALUDE_excess_amount_correct_l3821_382149

/-- The amount in excess of which the import tax is applied -/
def excess_amount : ℝ := 1000

/-- The total value of the item -/
def total_value : ℝ := 2560

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The amount of import tax paid -/
def tax_paid : ℝ := 109.20

/-- Theorem stating that the excess amount is correct given the conditions -/
theorem excess_amount_correct : 
  tax_rate * (total_value - excess_amount) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_excess_amount_correct_l3821_382149


namespace NUMINAMATH_CALUDE_eighth_term_is_22_n_equals_8_when_an_is_22_l3821_382188

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 8th term of the sequence is 22 -/
theorem eighth_term_is_22 : arithmeticSequence 8 = 22 := by
  sorry

/-- Theorem proving that if the nth term is 22, then n must be 8 -/
theorem n_equals_8_when_an_is_22 (n : ℕ) (h : arithmeticSequence n = 22) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_22_n_equals_8_when_an_is_22_l3821_382188


namespace NUMINAMATH_CALUDE_chrysanthemum_pots_count_l3821_382170

/-- The total number of chrysanthemum pots -/
def total_pots : ℕ := 360

/-- The number of rows after transportation -/
def remaining_rows : ℕ := 9

/-- The number of pots in each row -/
def pots_per_row : ℕ := 20

/-- Theorem stating that the total number of chrysanthemum pots is 360 -/
theorem chrysanthemum_pots_count :
  total_pots = 2 * remaining_rows * pots_per_row :=
by sorry

end NUMINAMATH_CALUDE_chrysanthemum_pots_count_l3821_382170


namespace NUMINAMATH_CALUDE_tan_22_5_deg_representation_l3821_382146

theorem tan_22_5_deg_representation :
  ∃ (a b c d : ℕ+), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) ∧
    (a + b + c + d = 10) := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_representation_l3821_382146


namespace NUMINAMATH_CALUDE_count_polynomials_l3821_382197

/-- A function to determine if an expression is a polynomial -/
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "x^2+2" => true
  | "1/a+4" => false
  | "3ab^2/7" => true
  | "ab/c" => false
  | "-5x" => true
  | "0" => true
  | _ => false

/-- The list of expressions to check -/
def expressions : List String := ["x^2+2", "1/a+4", "3ab^2/7", "ab/c", "-5x", "0"]

/-- Theorem stating that there are exactly 4 polynomial expressions in the given list -/
theorem count_polynomials : 
  (expressions.filter is_polynomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_polynomials_l3821_382197


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3821_382112

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, x > 2 → (x + 1) * (x - 2) > 0) ∧
  (∃ x : ℝ, (x + 1) * (x - 2) > 0 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3821_382112


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l3821_382136

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
  triangle_base = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l3821_382136


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3821_382140

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a - 4) + (b + 5)^2 = 0) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3821_382140


namespace NUMINAMATH_CALUDE_bag_cost_theorem_l3821_382143

def total_money : ℕ := 50
def tshirt_cost : ℕ := 8
def keychain_cost : ℚ := 2 / 3
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

theorem bag_cost_theorem :
  ∃ (bag_cost : ℚ),
    bag_cost * bags_bought = 
      total_money - 
      (tshirt_cost * tshirts_bought) - 
      (keychain_cost * keychains_bought) ∧
    bag_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_bag_cost_theorem_l3821_382143


namespace NUMINAMATH_CALUDE_total_letters_is_68_l3821_382169

/-- The total number of letters in all siblings' names -/
def total_letters : ℕ :=
  let jonathan_first := 8
  let jonathan_last := 10
  let younger_sister_first := 5
  let younger_sister_last := 10
  let older_brother_first := 6
  let older_brother_last := 10
  let youngest_sibling_first := 4
  let youngest_sibling_last := 15
  (jonathan_first + jonathan_last) +
  (younger_sister_first + younger_sister_last) +
  (older_brother_first + older_brother_last) +
  (youngest_sibling_first + youngest_sibling_last)

/-- Theorem stating that the total number of letters in all siblings' names is 68 -/
theorem total_letters_is_68 : total_letters = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_is_68_l3821_382169


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3821_382160

theorem algebraic_expression_value :
  let x : ℤ := -2
  let y : ℤ := -4
  2 * x^2 - y + 3 = 15 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3821_382160


namespace NUMINAMATH_CALUDE_final_stamp_collection_l3821_382105

def initial_stamps : ℕ := 3000
def mikes_gift : ℕ := 17
def damaged_stamps : ℕ := 37

def harrys_gift (mikes_gift : ℕ) : ℕ := 2 * mikes_gift + 10
def sarahs_gift (mikes_gift : ℕ) : ℕ := 3 * mikes_gift - 5

def total_gift_stamps (mikes_gift : ℕ) : ℕ :=
  mikes_gift + harrys_gift mikes_gift + sarahs_gift mikes_gift

def final_stamp_count (initial_stamps mikes_gift damaged_stamps : ℕ) : ℕ :=
  initial_stamps + total_gift_stamps mikes_gift - damaged_stamps

theorem final_stamp_collection :
  final_stamp_count initial_stamps mikes_gift damaged_stamps = 3070 :=
by sorry

end NUMINAMATH_CALUDE_final_stamp_collection_l3821_382105


namespace NUMINAMATH_CALUDE_abs_half_minus_three_eighths_i_equals_five_eighths_l3821_382178

theorem abs_half_minus_three_eighths_i_equals_five_eighths :
  Complex.abs (1/2 - 3/8 * Complex.I) = 5/8 := by sorry

end NUMINAMATH_CALUDE_abs_half_minus_three_eighths_i_equals_five_eighths_l3821_382178


namespace NUMINAMATH_CALUDE_doug_has_25_marbles_l3821_382139

/-- Calculates the number of marbles Doug has given the conditions of the problem. -/
def dougs_marbles (eds_initial_advantage : ℕ) (eds_lost_marbles : ℕ) (eds_current_marbles : ℕ) : ℕ :=
  eds_current_marbles + eds_lost_marbles - eds_initial_advantage

/-- Proves that Doug has 25 marbles given the conditions of the problem. -/
theorem doug_has_25_marbles :
  dougs_marbles 12 20 17 = 25 := by
  sorry

#eval dougs_marbles 12 20 17

end NUMINAMATH_CALUDE_doug_has_25_marbles_l3821_382139


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3821_382116

/-- 
Theorem: In a rhombus with side length 65 units and shorter diagonal 56 units, 
the longer diagonal is 118 units.
-/
theorem rhombus_longer_diagonal 
  (side_length : ℝ) 
  (shorter_diagonal : ℝ) 
  (h1 : side_length = 65) 
  (h2 : shorter_diagonal = 56) : ℝ :=
by
  -- Define the longer diagonal
  let longer_diagonal : ℝ := 118
  
  -- The proof would go here
  sorry

#check rhombus_longer_diagonal

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3821_382116


namespace NUMINAMATH_CALUDE_discount_calculation_l3821_382177

/-- Given a bill amount and a discount for double the time, calculate the discount for the original time. -/
theorem discount_calculation (bill_amount : ℝ) (double_time_discount : ℝ) 
  (h1 : bill_amount = 110) 
  (h2 : double_time_discount = 18.33) : 
  ∃ (original_discount : ℝ), original_discount = 9.165 ∧ 
  original_discount = double_time_discount / 2 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l3821_382177


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3821_382192

theorem quadratic_minimum (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 18 * x + 7
  ∀ y : ℝ, f x ≤ f y ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3821_382192


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_range_l3821_382161

theorem prime_quadratic_roots_range (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ x y : ℤ, x^2 + p*x - 520*p = 0 ∧ y^2 + p*y - 520*p = 0 ∧ x ≠ y) →
  11 < p ∧ p ≤ 21 :=
by sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_range_l3821_382161


namespace NUMINAMATH_CALUDE_candy_distribution_l3821_382141

theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 43)
  (h2 : pieces_per_student = 8) :
  num_students * pieces_per_student = 344 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3821_382141


namespace NUMINAMATH_CALUDE_total_pages_read_l3821_382151

-- Define reading speeds for each genre and focus level
def novel_speed : Fin 3 → ℕ
| 0 => 21  -- low focus
| 1 => 25  -- medium focus
| 2 => 30  -- high focus
| _ => 0

def graphic_novel_speed : Fin 3 → ℕ
| 0 => 30  -- low focus
| 1 => 36  -- medium focus
| 2 => 42  -- high focus
| _ => 0

def comic_book_speed : Fin 3 → ℕ
| 0 => 45  -- low focus
| 1 => 54  -- medium focus
| 2 => 60  -- high focus
| _ => 0

def non_fiction_speed : Fin 3 → ℕ
| 0 => 18  -- low focus
| 1 => 22  -- medium focus
| 2 => 28  -- high focus
| _ => 0

def biography_speed : Fin 3 → ℕ
| 0 => 20  -- low focus
| 1 => 24  -- medium focus
| 2 => 29  -- high focus
| _ => 0

-- Define time allocations for each hour
def hour1_allocation : List (ℕ × ℕ × ℕ) := [
  (20, 2, 0),  -- 20 minutes, high focus, novel
  (10, 0, 1),  -- 10 minutes, low focus, graphic novel
  (15, 1, 3),  -- 15 minutes, medium focus, non-fiction
  (15, 0, 4)   -- 15 minutes, low focus, biography
]

def hour2_allocation : List (ℕ × ℕ × ℕ) := [
  (25, 1, 2),  -- 25 minutes, medium focus, comic book
  (15, 2, 1),  -- 15 minutes, high focus, graphic novel
  (20, 0, 0)   -- 20 minutes, low focus, novel
]

def hour3_allocation : List (ℕ × ℕ × ℕ) := [
  (10, 2, 3),  -- 10 minutes, high focus, non-fiction
  (20, 1, 4),  -- 20 minutes, medium focus, biography
  (30, 0, 2)   -- 30 minutes, low focus, comic book
]

-- Function to calculate pages read for a given time, focus, and genre
def pages_read (time : ℕ) (focus : Fin 3) (genre : Fin 5) : ℚ :=
  let speed := match genre with
    | 0 => novel_speed focus
    | 1 => graphic_novel_speed focus
    | 2 => comic_book_speed focus
    | 3 => non_fiction_speed focus
    | 4 => biography_speed focus
    | _ => 0
  (time : ℚ) / 60 * speed

-- Function to calculate total pages read for a list of allocations
def total_pages (allocations : List (ℕ × ℕ × ℕ)) : ℚ :=
  allocations.foldl (fun acc (time, focus, genre) => acc + pages_read time ⟨focus, by sorry⟩ ⟨genre, by sorry⟩) 0

-- Theorem stating the total pages read
theorem total_pages_read :
  ⌊total_pages hour1_allocation + total_pages hour2_allocation + total_pages hour3_allocation⌋ = 100 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_read_l3821_382151


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l3821_382119

theorem chocolate_box_problem (C : ℕ) : 
  (C % 3 = 0) →  -- Total is divisible by 3
  (C / 3 - 6 + C / 3 - 7 + (C / 3 * 3 / 10 : ℕ) = 36) →  -- Remaining chocolates equation
  C = 63 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l3821_382119


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l3821_382194

open Complex

theorem magnitude_of_complex_fourth_power : 
  ‖(4 + 3 * Real.sqrt 3 * I : ℂ)^4‖ = 1849 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l3821_382194
