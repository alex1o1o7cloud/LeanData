import Mathlib

namespace worker_schedule_solution_correct_l2904_290463

/-- Represents the worker payment schedule problem over a 30-day period. -/
structure WorkerSchedule where
  total_days : ℕ
  daily_wage : ℕ
  daily_penalty : ℕ
  total_earnings : ℤ

/-- The solution to the worker schedule problem. -/
def solve_worker_schedule (ws : WorkerSchedule) : ℕ :=
  sorry

/-- Theorem stating the correctness of the solution for the given problem. -/
theorem worker_schedule_solution_correct (ws : WorkerSchedule) : 
  ws.total_days = 30 ∧ 
  ws.daily_wage = 100 ∧ 
  ws.daily_penalty = 25 ∧ 
  ws.total_earnings = 0 →
  solve_worker_schedule ws = 24 :=
sorry

end worker_schedule_solution_correct_l2904_290463


namespace julia_tag_game_l2904_290497

theorem julia_tag_game (tuesday_kids : ℕ) (monday_difference : ℕ) : 
  tuesday_kids = 5 → monday_difference = 1 → tuesday_kids + monday_difference = 6 :=
by sorry

end julia_tag_game_l2904_290497


namespace complex_fraction_pure_imaginary_l2904_290462

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((a + 3 * Complex.I) / (1 - Complex.I)) → a = 3 := by
  sorry

end complex_fraction_pure_imaginary_l2904_290462


namespace average_sum_is_six_l2904_290422

theorem average_sum_is_six (a b c d e : ℕ) (h : a + b + c + d + e > 0) :
  let teacher_avg := (5*a + 4*b + 3*c + 2*d + e) / (a + b + c + d + e)
  let kati_avg := (5*e + 4*d + 3*c + 2*b + a) / (a + b + c + d + e)
  teacher_avg + kati_avg = 6 := by
  sorry

end average_sum_is_six_l2904_290422


namespace food_allocation_l2904_290482

/-- Given a total budget allocated among three categories in a specific ratio,
    calculate the amount allocated to the second category. -/
def allocate_budget (total : ℚ) (ratio1 ratio2 ratio3 : ℕ) : ℚ :=
  (total * ratio2) / (ratio1 + ratio2 + ratio3)

/-- Theorem stating that given a total budget of 1800 allocated in the ratio 5:4:1,
    the amount allocated to the second category is 720. -/
theorem food_allocation :
  allocate_budget 1800 5 4 1 = 720 := by
  sorry

end food_allocation_l2904_290482


namespace no_max_value_cubic_l2904_290427

/-- The function f(x) = 3x^2 + 6x^3 + 27x + 100 has no maximum value over the real numbers -/
theorem no_max_value_cubic (x : ℝ) : 
  ¬∃ (M : ℝ), ∀ (x : ℝ), 3*x^2 + 6*x^3 + 27*x + 100 ≤ M :=
by sorry

end no_max_value_cubic_l2904_290427


namespace arithmetic_sqrt_of_nine_l2904_290411

theorem arithmetic_sqrt_of_nine (x : ℝ) :
  (x ≥ 0 ∧ x^2 = 9) → x = 3 := by sorry

end arithmetic_sqrt_of_nine_l2904_290411


namespace second_class_size_l2904_290437

def students_first_class : ℕ := 25
def avg_marks_first_class : ℚ := 50
def avg_marks_second_class : ℚ := 65
def avg_marks_all : ℚ := 59.23076923076923

theorem second_class_size :
  ∃ (x : ℕ), 
    (students_first_class * avg_marks_first_class + x * avg_marks_second_class) / (students_first_class + x) = avg_marks_all ∧
    x = 40 := by
  sorry

end second_class_size_l2904_290437


namespace f_3_is_even_l2904_290486

/-- Given a function f(x) = a(x-1)³ + bx + c where a is real and b, c are integers,
    if f(-1) = 2, then f(3) must be even. -/
theorem f_3_is_even (a : ℝ) (b c : ℤ) :
  let f : ℝ → ℝ := λ x => a * (x - 1)^3 + b * x + c
  (f (-1) = 2) → ∃ k : ℤ, f 3 = 2 * k := by
  sorry

end f_3_is_even_l2904_290486


namespace ducks_in_marsh_l2904_290460

/-- The number of ducks in a marsh, given the total number of birds and the number of geese. -/
def num_ducks (total_birds geese : ℕ) : ℕ := total_birds - geese

/-- Theorem stating that there are 37 ducks in the marsh. -/
theorem ducks_in_marsh : num_ducks 95 58 = 37 := by
  sorry

end ducks_in_marsh_l2904_290460


namespace samuel_breaks_two_cups_per_box_l2904_290468

theorem samuel_breaks_two_cups_per_box 
  (total_boxes : ℕ) 
  (pan_boxes : ℕ) 
  (cups_per_row : ℕ) 
  (rows_per_box : ℕ) 
  (remaining_cups : ℕ) 
  (h1 : total_boxes = 26)
  (h2 : pan_boxes = 6)
  (h3 : cups_per_row = 4)
  (h4 : rows_per_box = 5)
  (h5 : remaining_cups = 180) :
  let remaining_boxes := total_boxes - pan_boxes
  let decoration_boxes := remaining_boxes / 2
  let teacup_boxes := remaining_boxes - decoration_boxes
  let cups_per_box := cups_per_row * rows_per_box
  let total_cups := teacup_boxes * cups_per_box
  let broken_cups := total_cups - remaining_cups
  2 = broken_cups / teacup_boxes :=
by sorry

end samuel_breaks_two_cups_per_box_l2904_290468


namespace expression_evaluation_l2904_290400

theorem expression_evaluation :
  let x : ℝ := 3 + Real.sqrt 2
  (1 - 5 / (x + 2)) / ((x^2 - 6*x + 9) / (x + 2)) = Real.sqrt 2 / 2 := by
  sorry

end expression_evaluation_l2904_290400


namespace mariela_cards_total_l2904_290483

/-- Calculates the total number of cards Mariela received based on the given quantities -/
def total_cards (hospital_dozens : ℕ) (hospital_hundreds : ℕ) (home_dozens : ℕ) (home_hundreds : ℕ) : ℕ :=
  (hospital_dozens * 12 + hospital_hundreds * 100) + (home_dozens * 12 + home_hundreds * 100)

/-- Proves that Mariela received 1768 cards in total -/
theorem mariela_cards_total : total_cards 25 7 39 3 = 1768 := by
  sorry

end mariela_cards_total_l2904_290483


namespace sin_lt_tan_in_first_quadrant_half_angle_in_first_or_third_quadrant_sin_not_always_four_fifths_sector_angle_is_one_radian_l2904_290438

-- Statement ①
theorem sin_lt_tan_in_first_quadrant (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  Real.sin α < Real.tan α := by sorry

-- Statement ②
theorem half_angle_in_first_or_third_quadrant (α : Real) 
  (h : Real.pi / 2 < α ∧ α < Real.pi) :
  (0 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
  (Real.pi < α / 2 ∧ α / 2 < 3 * Real.pi / 2) := by sorry

-- Statement ③ (incorrect)
theorem sin_not_always_four_fifths (k : Real) (h : k ≠ 0) :
  ∃ α, Real.cos α = 3 * k / 5 ∧ Real.sin α = 4 * k / 5 ∧ Real.sin α ≠ 4 / 5 := by sorry

-- Statement ④
theorem sector_angle_is_one_radian (perimeter radius : Real) 
  (h1 : perimeter = 6) (h2 : radius = 2) :
  (perimeter - 2 * radius) / radius = 1 := by sorry

end sin_lt_tan_in_first_quadrant_half_angle_in_first_or_third_quadrant_sin_not_always_four_fifths_sector_angle_is_one_radian_l2904_290438


namespace half_abs_diff_squares_21_19_l2904_290403

theorem half_abs_diff_squares_21_19 : 
  (1 / 2 : ℝ) * |21^2 - 19^2| = 40 := by sorry

end half_abs_diff_squares_21_19_l2904_290403


namespace faye_coloring_books_l2904_290473

theorem faye_coloring_books (initial : ℝ) (given_away : ℝ) (additional_percentage : ℝ) :
  initial = 52.5 →
  given_away = 38.2 →
  additional_percentage = 25 →
  let remainder : ℝ := initial - given_away
  let additional_given : ℝ := (additional_percentage / 100) * remainder
  initial - given_away - additional_given = 10.725 := by
  sorry

end faye_coloring_books_l2904_290473


namespace project_scientists_l2904_290443

/-- The total number of scientists in the project -/
def S : ℕ := 70

/-- The number of scientists from Europe -/
def europe : ℕ := S / 2

/-- The number of scientists from Canada -/
def canada : ℕ := S / 5

/-- The number of scientists from the USA -/
def usa : ℕ := 21

/-- Theorem stating that the sum of scientists from Europe, Canada, and USA equals the total number of scientists -/
theorem project_scientists : europe + canada + usa = S := by sorry

end project_scientists_l2904_290443


namespace difference_of_squares_application_l2904_290490

theorem difference_of_squares_application (a b : ℝ) :
  (1/4 * a + b) * (b - 1/4 * a) = b^2 - (1/16) * a^2 := by
  sorry

end difference_of_squares_application_l2904_290490


namespace mass_percentage_cl_in_mixture_mass_percentage_cl_approx_43_85_l2904_290431

/-- Mass percentage of Cl in a mixture of NaClO and NaClO2 -/
theorem mass_percentage_cl_in_mixture (moles_NaClO moles_NaClO2 : ℝ) 
  (mass_Na mass_Cl mass_O : ℝ) : ℝ :=
  let molar_mass_NaClO := mass_Na + mass_Cl + mass_O
  let molar_mass_NaClO2 := mass_Na + mass_Cl + 2 * mass_O
  let mass_Cl_NaClO := moles_NaClO * mass_Cl
  let mass_Cl_NaClO2 := moles_NaClO2 * mass_Cl
  let total_mass_Cl := mass_Cl_NaClO + mass_Cl_NaClO2
  let total_mass_mixture := moles_NaClO * molar_mass_NaClO + moles_NaClO2 * molar_mass_NaClO2
  let mass_percentage_Cl := (total_mass_Cl / total_mass_mixture) * 100
  mass_percentage_Cl

/-- The mass percentage of Cl in the given mixture is approximately 43.85% -/
theorem mass_percentage_cl_approx_43_85 :
  abs (mass_percentage_cl_in_mixture 3 2 22.99 35.45 16 - 43.85) < 0.01 :=
sorry

end mass_percentage_cl_in_mixture_mass_percentage_cl_approx_43_85_l2904_290431


namespace complex_exponential_185_54_l2904_290459

theorem complex_exponential_185_54 :
  (Complex.exp (185 * Real.pi / 180 * Complex.I))^54 = -Complex.I := by
  sorry

end complex_exponential_185_54_l2904_290459


namespace total_bugs_is_63_l2904_290441

/-- The number of bugs eaten by the gecko, lizard, frog, and toad -/
def total_bugs_eaten (gecko_bugs : ℕ) : ℕ :=
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + frog_bugs / 2
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs

/-- Theorem stating the total number of bugs eaten is 63 -/
theorem total_bugs_is_63 : total_bugs_eaten 12 = 63 := by
  sorry

#eval total_bugs_eaten 12

end total_bugs_is_63_l2904_290441


namespace triangle_perpendicular_theorem_l2904_290487

structure Triangle (P Q R : ℝ × ℝ) where
  pq_length : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 15
  pr_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 20

def foot_of_perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (S.1 - Q.1) * (R.1 - Q.1) + (S.2 - Q.2) * (R.2 - Q.2) = 0 ∧
  (P.1 - S.1) * (R.1 - Q.1) + (P.2 - S.2) * (R.2 - Q.2) = 0

def segment_ratio (Q S R : ℝ × ℝ) : Prop :=
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) / Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 3 / 7

theorem triangle_perpendicular_theorem (P Q R S : ℝ × ℝ) 
  (tri : Triangle P Q R) (foot : foot_of_perpendicular P Q R S) (ratio : segment_ratio Q S R) :
  Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 13.625 := by
  sorry

end triangle_perpendicular_theorem_l2904_290487


namespace money_distribution_l2904_290449

/-- Given three people A, B, and C with money, prove that B and C together have 350 rupees. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 450 →  -- Total money
  a + c = 200 →      -- Money A and C have together
  c = 100 →          -- Money C has
  b + c = 350 :=     -- Money B and C have together
by
  sorry

end money_distribution_l2904_290449


namespace gasoline_added_l2904_290495

theorem gasoline_added (tank_capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : tank_capacity = 54 → initial_fill = 3/4 → final_fill = 9/10 → (final_fill - initial_fill) * tank_capacity = 8.1 := by
  sorry

end gasoline_added_l2904_290495


namespace pizza_combinations_l2904_290407

theorem pizza_combinations : Nat.choose 8 5 = 56 := by
  sorry

end pizza_combinations_l2904_290407


namespace green_paint_amount_l2904_290436

/-- The amount of green paint needed for a treehouse project. -/
def green_paint (total white brown : ℕ) : ℕ :=
  total - (white + brown)

/-- Theorem stating that the amount of green paint is 15 ounces. -/
theorem green_paint_amount :
  green_paint 69 20 34 = 15 := by
  sorry

end green_paint_amount_l2904_290436


namespace octagon_quad_area_ratio_l2904_290496

/-- Regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Quadrilateral formed by connecting alternate vertices of the octagon -/
def alternateVerticesQuad (octagon : RegularOctagon) : Fin 4 → ℝ × ℝ :=
  fun i => octagon.vertices (2 * i)

/-- Area of a polygon given its vertices -/
def polygonArea (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

theorem octagon_quad_area_ratio 
  (octagon : RegularOctagon) 
  (n : ℝ) 
  (m : ℝ) 
  (hn : n = polygonArea octagon.vertices) 
  (hm : m = polygonArea (alternateVerticesQuad octagon)) :
  m / n = Real.sqrt 2 / 2 :=
sorry

end octagon_quad_area_ratio_l2904_290496


namespace expression_value_l2904_290405

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : abs m = 3)  -- |m| = 3
  : m + c * d - (a + b) / (m^2) = 4 ∨ m + c * d - (a + b) / (m^2) = -2 := by
  sorry

end expression_value_l2904_290405


namespace inequality_proof_l2904_290414

open Real BigOperators Finset

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (σ : Equiv.Perm (Fin n)) 
  (h : ∀ i, 0 < x i ∧ x i < 1) : 
  ∑ i, (1 / (1 - x i)) ≥ 
  (1 + (1 / n) * ∑ i, x i) * ∑ i, (1 / (1 - x i * x (σ i))) := by
  sorry

end inequality_proof_l2904_290414


namespace happy_children_count_l2904_290419

theorem happy_children_count (total_children : ℕ) 
                              (sad_children : ℕ) 
                              (neutral_children : ℕ) 
                              (total_boys : ℕ) 
                              (total_girls : ℕ) 
                              (happy_boys : ℕ) 
                              (sad_girls : ℕ) :
  total_children = 60 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 18 →
  total_girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  ∃ (happy_children : ℕ), 
    happy_children = 30 ∧
    happy_children + sad_children + neutral_children = total_children ∧
    happy_boys + (sad_children - sad_girls) + (neutral_children - (neutral_children - (total_boys - happy_boys - (sad_children - sad_girls)))) = total_boys ∧
    (happy_children - happy_boys) + sad_girls + (neutral_children - (total_boys - happy_boys - (sad_children - sad_girls))) = total_girls :=
by
  sorry


end happy_children_count_l2904_290419


namespace certain_number_proof_l2904_290475

theorem certain_number_proof (N : ℝ) : (5/6) * N = (5/16) * N + 50 → N = 96 := by
  sorry

end certain_number_proof_l2904_290475


namespace inequality_solution_set_l2904_290415

theorem inequality_solution_set (a : ℝ) :
  (∀ x, (x - a) * (x + a - 1) > 0 ↔ 
    (a = 1/2 ∧ x ≠ 1/2) ∨
    (a < 1/2 ∧ (x > 1 - a ∨ x < a)) ∨
    (a > 1/2 ∧ (x > a ∨ x < 1 - a))) :=
by sorry

end inequality_solution_set_l2904_290415


namespace swim_time_ratio_l2904_290406

/-- Proves that the ratio of time taken to swim upstream to downstream is 2:1 given specific speeds -/
theorem swim_time_ratio (man_speed stream_speed : ℝ) 
  (h1 : man_speed = 3)
  (h2 : stream_speed = 1) :
  (man_speed - stream_speed)⁻¹ / (man_speed + stream_speed)⁻¹ = 2 := by
  sorry

end swim_time_ratio_l2904_290406


namespace initial_deposit_calculation_l2904_290478

/-- Proves that the initial deposit is $1000 given the conditions of the problem -/
theorem initial_deposit_calculation (P : ℝ) : 
  (P + 100 = 1100) →                    -- First year balance
  ((P + 100) * 1.2 = P * 1.32) →         -- Second year growth equals total growth
  P = 1000 := by
  sorry

end initial_deposit_calculation_l2904_290478


namespace snow_leopard_arrangements_l2904_290434

/-- The number of ways to arrange n different objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of snow leopards --/
def total_leopards : ℕ := 8

/-- The number of leopards that can be freely arranged --/
def free_leopards : ℕ := total_leopards - 2

/-- The number of ways to arrange the shortest and tallest leopards --/
def end_arrangements : ℕ := 2

theorem snow_leopard_arrangements :
  end_arrangements * permutations free_leopards = 1440 := by
  sorry

end snow_leopard_arrangements_l2904_290434


namespace chess_tournament_attendance_l2904_290446

theorem chess_tournament_attendance (total_students : ℕ) 
  (h1 : total_students = 24) 
  (h2 : ∃ chess_students : ℕ, chess_students = total_students / 3)
  (h3 : ∃ tournament_students : ℕ, tournament_students = (total_students / 3) / 2) :
  ∃ tournament_students : ℕ, tournament_students = 4 := by
sorry

end chess_tournament_attendance_l2904_290446


namespace specific_right_triangle_with_square_l2904_290420

/-- Represents a right triangle with a square inscribed on its hypotenuse -/
structure RightTriangleWithSquare where
  /-- Length of one leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- Distance from the right angle vertex to the side of the square on the hypotenuse -/
  distance_to_square : ℝ

/-- Theorem stating the properties of the specific right triangle with inscribed square -/
theorem specific_right_triangle_with_square :
  ∃ (t : RightTriangleWithSquare),
    t.leg1 = 9 ∧
    t.leg2 = 12 ∧
    t.square_side = 75 / 7 ∧
    t.distance_to_square = 36 / 5 := by
  sorry

end specific_right_triangle_with_square_l2904_290420


namespace initial_salt_concentration_l2904_290481

theorem initial_salt_concentration
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_salt_percentage : ℝ)
  (h1 : initial_volume = 56)
  (h2 : water_added = 14)
  (h3 : final_salt_percentage = 0.08)
  (h4 : initial_volume * initial_salt_percentage = (initial_volume + water_added) * final_salt_percentage) :
  initial_salt_percentage = 0.1 := by
  sorry

end initial_salt_concentration_l2904_290481


namespace P_roots_properties_l2904_290444

/-- Definition of the polynomial sequence P_n(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(5*(n+1)) - P n x

/-- Theorem stating the properties of real roots for P_n(x) -/
theorem P_roots_properties :
  (∀ n : ℕ, Odd n → (∃! x : ℝ, P n x = 0 ∧ x = 1)) ∧
  (∀ n : ℕ, Even n → ∀ x : ℝ, P n x ≠ 0) :=
sorry

end P_roots_properties_l2904_290444


namespace rachel_painting_time_l2904_290429

/-- Prove that Rachel's painting time is 13 hours -/
theorem rachel_painting_time :
  let matt_time : ℕ := 12
  let patty_time : ℕ := matt_time / 3
  let rachel_time : ℕ := 2 * patty_time + 5
  rachel_time = 13 := by
  sorry

end rachel_painting_time_l2904_290429


namespace negation_of_universal_proposition_l2904_290480

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by sorry

end negation_of_universal_proposition_l2904_290480


namespace negative_integer_solution_of_inequality_l2904_290409

theorem negative_integer_solution_of_inequality :
  ∀ x : ℤ, x < 0 →
    (((2 * x - 1 : ℚ) / 3) - ((5 * x + 1 : ℚ) / 2) ≤ 1) ↔ x = -1 := by
  sorry

end negative_integer_solution_of_inequality_l2904_290409


namespace line_parallel_to_plane_l2904_290402

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular α β) 
  (h2 : line_perpendicular m β) 
  (h3 : ¬ line_in_plane m α) : 
  line_parallel m α :=
sorry

end line_parallel_to_plane_l2904_290402


namespace factorization_x_squared_minus_2x_l2904_290479

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end factorization_x_squared_minus_2x_l2904_290479


namespace jasmine_buys_six_bags_l2904_290401

/-- The number of bags of chips Jasmine buys -/
def bags_of_chips : ℕ := sorry

/-- The weight of one bag of chips in ounces -/
def chips_weight : ℕ := 20

/-- The weight of one tin of cookies in ounces -/
def cookies_weight : ℕ := 9

/-- The total weight Jasmine carries in ounces -/
def total_weight : ℕ := 21 * 16

theorem jasmine_buys_six_bags :
  bags_of_chips = 6 ∧
  chips_weight * bags_of_chips + cookies_weight * (4 * bags_of_chips) = total_weight :=
by sorry

end jasmine_buys_six_bags_l2904_290401


namespace can_reach_ten_white_marbles_l2904_290491

-- Define the state of the urn
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

-- Define the possible operations
inductive Operation
  | op1 -- 4B -> 2B
  | op2 -- 3B + W -> B
  | op3 -- 2B + 2W -> W + B
  | op4 -- B + 3W -> 2W
  | op5 -- 4W -> B

-- Define a function to apply an operation to the urn state
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 2⟩
  | Operation.op2 => ⟨state.white - 1, state.black - 2⟩
  | Operation.op3 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op4 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op5 => ⟨state.white - 4, state.black + 1⟩

-- Define the initial state
def initialState : UrnState := ⟨50, 150⟩

-- Theorem: It is possible to reach exactly 10 white marbles
theorem can_reach_ten_white_marbles :
  ∃ (operations : List Operation),
    (operations.foldl applyOperation initialState).white = 10 :=
sorry

end can_reach_ten_white_marbles_l2904_290491


namespace arithmetic_sequence_sum_l2904_290412

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (a 1 + a n) * n / 2) →   -- sum formula for arithmetic sequence
  (a 2 - 1)^3 + 2014 * (a 2 - 1) = Real.sin (2011 * Real.pi / 3) →
  (a 2013 - 1)^3 + 2014 * (a 2013 - 1) = Real.cos (2011 * Real.pi / 6) →
  S 2014 = 2014 := by
  sorry

end arithmetic_sequence_sum_l2904_290412


namespace sum_of_four_unit_fractions_l2904_290465

theorem sum_of_four_unit_fractions : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1 :=
by
  -- Proof goes here
  sorry

end sum_of_four_unit_fractions_l2904_290465


namespace max_value_quadratic_l2904_290435

theorem max_value_quadratic (x : ℝ) : 
  ∃ (max : ℝ), max = 9 ∧ ∀ y : ℝ, y = -3 * x^2 + 9 → y ≤ max :=
sorry

end max_value_quadratic_l2904_290435


namespace nine_point_circle_triangles_l2904_290452

/-- Given 9 points on a circle, this function calculates the number of distinct triangles
    formed by the intersection points of chords inside the circle. --/
def count_triangles (n : ℕ) : ℕ :=
  Nat.choose n 6

/-- Theorem stating that for 9 points on a circle, with chords connecting every pair of points
    and no three chords intersecting at a single point inside the circle, the number of
    distinct triangles formed by the intersection points of these chords inside the circle is 84. --/
theorem nine_point_circle_triangles :
  count_triangles 9 = 84 := by
  sorry

end nine_point_circle_triangles_l2904_290452


namespace wood_length_after_sawing_l2904_290426

theorem wood_length_after_sawing (original_length saw_length : Real) 
  (h1 : original_length = 0.41)
  (h2 : saw_length = 0.33) :
  original_length - saw_length = 0.08 := by
  sorry

end wood_length_after_sawing_l2904_290426


namespace trip_cost_calculation_l2904_290494

theorem trip_cost_calculation (original_price discount : ℕ) (num_people : ℕ) : 
  original_price = 147 → 
  discount = 14 → 
  num_people = 2 → 
  (original_price - discount) * num_people = 266 := by
sorry

end trip_cost_calculation_l2904_290494


namespace adam_tickets_bought_l2904_290498

def tickets_bought (tickets_left : ℕ) (ticket_cost : ℕ) (amount_spent : ℕ) : ℕ :=
  tickets_left + amount_spent / ticket_cost

theorem adam_tickets_bought :
  tickets_bought 4 9 81 = 13 := by
  sorry

end adam_tickets_bought_l2904_290498


namespace notebook_cost_l2904_290454

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 35 ∧
  total_cost = 2013 ∧
  buyers > total_students / 2 ∧
  notebooks_per_student % 2 = 0 ∧
  notebooks_per_student > 2 ∧
  cost_per_notebook > notebooks_per_student ∧
  buyers * notebooks_per_student * cost_per_notebook = total_cost ∧
  cost_per_notebook = 61 :=
by sorry

end notebook_cost_l2904_290454


namespace probability_of_losing_l2904_290451

theorem probability_of_losing (odds_win odds_lose : ℕ) 
  (h_odds : odds_win = 5 ∧ odds_lose = 3) : 
  (odds_lose : ℚ) / (odds_win + odds_lose) = 3 / 8 :=
by
  sorry

#check probability_of_losing

end probability_of_losing_l2904_290451


namespace last_two_digits_product_l2904_290442

def last_two_digits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

theorem last_two_digits_product (n : ℤ) : 
  n % 4 = 0 → 
  (let (a, b) := last_two_digits n; a + b = 12) → 
  (let (a, b) := last_two_digits n; a * b = 32 ∨ a * b = 36) :=
by sorry

end last_two_digits_product_l2904_290442


namespace smallest_x_for_equation_l2904_290417

theorem smallest_x_for_equation : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∃ (y : ℕ), y > 0 ∧ (0.8 : ℚ) = y / (196 + x)) ∧
  (∀ (x' : ℕ), x' > 0 → x' < x → 
    ¬∃ (y : ℕ), y > 0 ∧ (0.8 : ℚ) = y / (196 + x')) ∧
  x = 49 := by
sorry

end smallest_x_for_equation_l2904_290417


namespace power_function_value_l2904_290445

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f (1/2) = 8 → f 2 = 1/8 := by
  sorry

end power_function_value_l2904_290445


namespace isosceles_triangle_perimeter_l2904_290493

-- Define the sides of the triangle
def side1 : ℝ := 9
def side2 : ℝ := 9
def side3 : ℝ := 4

-- Define the isosceles triangle condition
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  is_isosceles side1 side2 side3 ∧
  satisfies_triangle_inequality side1 side2 side3 →
  perimeter side1 side2 side3 = 22 :=
by sorry

end isosceles_triangle_perimeter_l2904_290493


namespace angle_inequality_l2904_290469

theorem angle_inequality (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → 
    x^2 * Real.sin θ - x * (2 - x) + (2 - x)^2 * Real.cos θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) := by
  sorry

end angle_inequality_l2904_290469


namespace midpoint_distance_after_movement_l2904_290484

/-- Given two points A and B in a Cartesian plane, if A moves 5 units right and 6 units up,
    and B moves 12 units left and 4 units down, then the distance between the original
    midpoint M and the new midpoint M' is √53/2. -/
theorem midpoint_distance_after_movement (p q r s : ℝ) : 
  let A : ℝ × ℝ := (p, q)
  let B : ℝ × ℝ := (r, s)
  let M : ℝ × ℝ := ((p + r) / 2, (q + s) / 2)
  let A' : ℝ × ℝ := (p + 5, q + 6)
  let B' : ℝ × ℝ := (r - 12, s - 4)
  let M' : ℝ × ℝ := ((p + 5 + r - 12) / 2, (q + 6 + s - 4) / 2)
  Real.sqrt ((M.1 - M'.1)^2 + (M.2 - M'.2)^2) = Real.sqrt 53 / 2 := by
  sorry

end midpoint_distance_after_movement_l2904_290484


namespace ladder_distance_l2904_290410

theorem ladder_distance (ladder_length height : ℝ) 
  (h1 : ladder_length = 25)
  (h2 : height = 20) :
  ∃ (distance : ℝ), distance^2 + height^2 = ladder_length^2 ∧ distance = 15 :=
sorry

end ladder_distance_l2904_290410


namespace camp_gender_difference_l2904_290455

theorem camp_gender_difference (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 133 →
  girls = 50 →
  boys > girls →
  total = boys + girls →
  boys - girls = 33 := by
sorry

end camp_gender_difference_l2904_290455


namespace units_digit_not_zero_l2904_290428

theorem units_digit_not_zero (a b : Nat) (ha : a ∈ Finset.range 100) (hb : b ∈ Finset.range 100) :
  (5^a + 6^b) % 10 ≠ 0 := by
sorry

end units_digit_not_zero_l2904_290428


namespace fourth_month_sale_is_13792_l2904_290458

/-- Represents the sales data for a grocery shop over 6 months -/
structure SalesData where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the sale in the fourth month given the sales data and average -/
def fourthMonthSale (data : SalesData) (average : ℕ) : ℕ :=
  6 * average - (data.month1 + data.month2 + data.month3 + data.month5 + data.month6)

/-- Theorem stating that the fourth month's sale is 13792 given the conditions -/
theorem fourth_month_sale_is_13792 :
  let data : SalesData := {
    month1 := 6635,
    month2 := 6927,
    month3 := 6855,
    month4 := 0,  -- Unknown, to be calculated
    month5 := 6562,
    month6 := 4791
  }
  let average := 6500
  fourthMonthSale data average = 13792 := by
  sorry

#eval fourthMonthSale
  { month1 := 6635,
    month2 := 6927,
    month3 := 6855,
    month4 := 0,
    month5 := 6562,
    month6 := 4791 }
  6500

end fourth_month_sale_is_13792_l2904_290458


namespace impossible_sum_of_two_smaller_angles_l2904_290474

theorem impossible_sum_of_two_smaller_angles (α β γ : ℝ) : 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = 180 → 
  α ≤ γ → β ≤ γ → 
  α + β ≠ 130 :=
sorry

end impossible_sum_of_two_smaller_angles_l2904_290474


namespace min_perimeter_rectangle_l2904_290457

theorem min_perimeter_rectangle (w l : ℝ) (h1 : w > 0) (h2 : l > 0) (h3 : l = 2 * w) (h4 : w * l ≥ 500) :
  2 * w + 2 * l ≥ 30 * Real.sqrt 10 ∧ 
  (2 * w + 2 * l = 30 * Real.sqrt 10 → w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10) := by
  sorry

end min_perimeter_rectangle_l2904_290457


namespace no_linear_term_condition_l2904_290492

theorem no_linear_term_condition (p q : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x^2 - p*x + q)*(x - 3) = a*x^3 + b*x^2 + c) → 
  q + 3*p = 0 := by
sorry

end no_linear_term_condition_l2904_290492


namespace atlantic_call_rate_l2904_290466

/-- Proves that the additional charge per minute for Atlantic Call is $0.20 -/
theorem atlantic_call_rate (united_base_rate : ℚ) (united_per_minute : ℚ) 
  (atlantic_base_rate : ℚ) (minutes : ℕ) :
  united_base_rate = 7 →
  united_per_minute = 0.25 →
  atlantic_base_rate = 12 →
  minutes = 100 →
  united_base_rate + united_per_minute * minutes = 
    atlantic_base_rate + (atlantic_base_rate + united_per_minute * minutes - united_base_rate) / minutes →
  (atlantic_base_rate + united_per_minute * minutes - united_base_rate) / minutes = 0.20 := by
  sorry

#check atlantic_call_rate

end atlantic_call_rate_l2904_290466


namespace line_perp_plane_condition_l2904_290425

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the relation of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- The theorem statement
theorem line_perp_plane_condition 
  (m n : Line) (α β : Plane) 
  (h1 : perp_planes α β)
  (h2 : intersect α β = m)
  (h3 : contained_in n α) :
  perp_line_plane n β ↔ perp_lines n m :=
sorry

end line_perp_plane_condition_l2904_290425


namespace new_person_weight_l2904_290499

theorem new_person_weight (n : ℕ) (old_weight avg_increase : ℝ) :
  n = 8 ∧ 
  old_weight = 70 ∧ 
  avg_increase = 3 →
  (n * avg_increase + old_weight : ℝ) = 94 := by
  sorry

end new_person_weight_l2904_290499


namespace total_spent_is_450_l2904_290430

/-- The total amount spent by Leonard and Michael on presents for their father -/
def total_spent (leonard_wallet : ℕ) (leonard_sneakers : ℕ) (leonard_sneakers_pairs : ℕ)
  (michael_backpack : ℕ) (michael_jeans : ℕ) (michael_jeans_pairs : ℕ) : ℕ :=
  leonard_wallet + leonard_sneakers * leonard_sneakers_pairs +
  michael_backpack + michael_jeans * michael_jeans_pairs

/-- Theorem stating that the total amount spent is $450 -/
theorem total_spent_is_450 :
  total_spent 50 100 2 100 50 2 = 450 := by
  sorry

end total_spent_is_450_l2904_290430


namespace mark_bench_press_value_l2904_290472

/-- Dave's weight in pounds -/
def dave_weight : ℝ := 175

/-- Dave's bench press multiplier -/
def dave_multiplier : ℝ := 3

/-- Craig's bench press percentage compared to Dave -/
def craig_percentage : ℝ := 0.2

/-- Difference between Craig's and Mark's bench press in pounds -/
def mark_difference : ℝ := 50

/-- Calculate Dave's bench press weight -/
def dave_bench_press : ℝ := dave_weight * dave_multiplier

/-- Calculate Craig's bench press weight -/
def craig_bench_press : ℝ := dave_bench_press * craig_percentage

/-- Calculate Mark's bench press weight -/
def mark_bench_press : ℝ := craig_bench_press - mark_difference

theorem mark_bench_press_value : mark_bench_press = 55 := by
  sorry

end mark_bench_press_value_l2904_290472


namespace min_value_of_f_l2904_290418

/-- The function we want to minimize -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ 0) ∧ f (-1) 3 = 0 := by
  sorry

end min_value_of_f_l2904_290418


namespace polynomial_remainder_l2904_290440

theorem polynomial_remainder (x : ℤ) : (x^15 - 2) % (x + 2) = -32770 := by
  sorry

end polynomial_remainder_l2904_290440


namespace initial_items_count_l2904_290485

/-- The number of items Adam initially put in the shopping cart -/
def initial_items : ℕ := sorry

/-- The number of items Adam deleted from the shopping cart -/
def deleted_items : ℕ := 10

/-- The number of items left in Adam's shopping cart after deletion -/
def remaining_items : ℕ := 8

/-- Theorem stating that the initial number of items is 18 -/
theorem initial_items_count : initial_items = 18 :=
  by sorry

end initial_items_count_l2904_290485


namespace ellipse_max_product_l2904_290423

theorem ellipse_max_product (x y : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  (x^2 / 25 + y^2 / 9 = 1) →
  (P = (x, y)) →
  (F₁ ≠ F₂) →
  (∀ (x' y' : ℝ), x'^2 / 25 + y'^2 / 9 = 1 → 
    dist P F₁ + dist P F₂ = dist (x', y') F₁ + dist (x', y') F₂) →
  (∃ (M : ℝ), ∀ (x' y' : ℝ), x'^2 / 25 + y'^2 / 9 = 1 → 
    dist (x', y') F₁ * dist (x', y') F₂ ≤ M ∧ 
    ∃ (x'' y'' : ℝ), x''^2 / 25 + y''^2 / 9 = 1 ∧ 
      dist (x'', y'') F₁ * dist (x'', y'') F₂ = M) →
  M = 25 := by
sorry

end ellipse_max_product_l2904_290423


namespace class_size_l2904_290464

theorem class_size (hockey : ℕ) (basketball : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : hockey = 15)
  (h2 : basketball = 16)
  (h3 : both = 10)
  (h4 : neither = 4) :
  hockey + basketball - both + neither = 25 := by
  sorry

end class_size_l2904_290464


namespace total_spent_on_tickets_l2904_290476

def this_year_prices : List ℕ := [35, 45, 50, 62]
def last_year_prices : List ℕ := [25, 30, 40, 45, 55, 60, 65, 70, 75]

theorem total_spent_on_tickets : 
  (this_year_prices.sum + last_year_prices.sum : ℕ) = 657 := by
  sorry

end total_spent_on_tickets_l2904_290476


namespace integer_part_of_sum_of_roots_l2904_290489

theorem integer_part_of_sum_of_roots (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x*y + y*z + z*x = 1) : 
  ⌊Real.sqrt (3*x*y + 1) + Real.sqrt (3*y*z + 1) + Real.sqrt (3*z*x + 1)⌋ = 4 :=
sorry

end integer_part_of_sum_of_roots_l2904_290489


namespace dot_only_count_l2904_290424

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (dot_and_line : ℕ)
  (line_only : ℕ)
  (all_have_dot_or_line : Prop)

/-- The number of letters containing a dot but not a straight line -/
def dot_only (α : Alphabet) : ℕ :=
  α.total - α.dot_and_line - α.line_only

/-- Theorem stating the number of letters with only a dot in the given alphabet -/
theorem dot_only_count (α : Alphabet) 
  (h1 : α.total = 40)
  (h2 : α.dot_and_line = 13)
  (h3 : α.line_only = 24) :
  dot_only α = 16 := by
  sorry

end dot_only_count_l2904_290424


namespace floor_negative_seven_fourths_l2904_290439

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end floor_negative_seven_fourths_l2904_290439


namespace f_not_in_second_quadrant_l2904_290461

/-- A linear function f(x) = 2x - 1 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of f(x) = 2x - 1 does not pass through the second quadrant -/
theorem f_not_in_second_quadrant :
  ∀ x y : ℝ, f x = y → ¬(second_quadrant x y) :=
by sorry

end f_not_in_second_quadrant_l2904_290461


namespace jia_steps_to_meet_yi_l2904_290421

theorem jia_steps_to_meet_yi (distance : ℝ) (speed_ratio : ℝ) (step_length : ℝ) :
  distance = 10560 ∧ speed_ratio = 5 ∧ step_length = 2.5 →
  (distance / (1 + speed_ratio)) / step_length = 704 := by
  sorry

end jia_steps_to_meet_yi_l2904_290421


namespace nested_radical_equality_l2904_290488

theorem nested_radical_equality : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end nested_radical_equality_l2904_290488


namespace count_symmetric_scanning_codes_l2904_290432

/-- A symmetric scanning code is a 7x7 grid of black and white squares that is invariant under 90° rotations and reflections across diagonals and midlines. -/
def SymmetricScanningCode := Fin 7 → Fin 7 → Bool

/-- A scanning code is valid if it has at least one black and one white square. -/
def is_valid (code : SymmetricScanningCode) : Prop :=
  (∃ i j, code i j = true) ∧ (∃ i j, code i j = false)

/-- A scanning code is symmetric if it's invariant under 90° rotations and reflections. -/
def is_symmetric (code : SymmetricScanningCode) : Prop :=
  (∀ i j, code i j = code (6-j) i) ∧  -- 90° rotation
  (∀ i j, code i j = code j i) ∧      -- diagonal reflection
  (∀ i j, code i j = code (6-i) (6-j))  -- midline reflection

/-- The number of valid symmetric scanning codes -/
def num_valid_symmetric_codes : ℕ := sorry

theorem count_symmetric_scanning_codes :
  num_valid_symmetric_codes = 1022 :=
sorry

end count_symmetric_scanning_codes_l2904_290432


namespace arithmetic_sequence_property_l2904_290448

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  n : ℕ -- number of terms
  d : ℝ -- common difference
  a₁ : ℝ -- first term

/-- Sum of magnitudes of terms in an arithmetic sequence -/
def sumOfMagnitudes (seq : ArithmeticSequence) : ℝ := 
  sorry

/-- New sequence obtained by adding a constant to all terms -/
def addConstant (seq : ArithmeticSequence) (c : ℝ) : ArithmeticSequence :=
  sorry

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sumOfMagnitudes seq = 250 ∧
  sumOfMagnitudes (addConstant seq 1) = 250 ∧
  sumOfMagnitudes (addConstant seq 2) = 250 →
  seq.n^2 * seq.d = 1000 ∨ seq.n^2 * seq.d = -1000 := by
  sorry

end arithmetic_sequence_property_l2904_290448


namespace inequality_integer_solutions_l2904_290450

theorem inequality_integer_solutions :
  {x : ℤ | 3 ≤ 5 - 2*x ∧ 5 - 2*x ≤ 9} = {-2, -1, 0, 1} := by
  sorry

end inequality_integer_solutions_l2904_290450


namespace trig_problem_l2904_290456

theorem trig_problem (a : Real) (h1 : 0 < a) (h2 : a < Real.pi) (h3 : Real.tan a = -2) :
  (Real.cos a = -Real.sqrt 5 / 5) ∧
  (2 * Real.sin a ^ 2 - Real.sin a * Real.cos a + Real.cos a ^ 2 = 11 / 5) := by
  sorry

end trig_problem_l2904_290456


namespace complex_fraction_simplification_l2904_290433

theorem complex_fraction_simplification :
  (7 + 15 * Complex.I) / (3 - 4 * Complex.I) = -39/25 + (73/25) * Complex.I :=
by sorry

end complex_fraction_simplification_l2904_290433


namespace sqrt_diff_inequality_l2904_290447

theorem sqrt_diff_inequality (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n - 1) - Real.sqrt n < Real.sqrt n - Real.sqrt (n + 1) :=
by sorry

end sqrt_diff_inequality_l2904_290447


namespace prime_sequence_l2904_290470

theorem prime_sequence (A : ℕ) : 
  Nat.Prime A ∧ 
  Nat.Prime (A + 14) ∧ 
  Nat.Prime (A + 18) ∧ 
  Nat.Prime (A + 32) ∧ 
  Nat.Prime (A + 36) → 
  A = 5 :=
by sorry

end prime_sequence_l2904_290470


namespace student_927_selected_l2904_290477

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : Nat := 1000

/-- The number of students to be sampled -/
def sampleSize : Nat := 200

/-- The sampling interval -/
def samplingInterval : Nat := totalStudents / sampleSize

/-- Predicate to check if a student number is selected in the systematic sampling -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % samplingInterval = 122 % samplingInterval

/-- Theorem stating that if student 122 is selected, then student 927 is also selected -/
theorem student_927_selected :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end student_927_selected_l2904_290477


namespace square_side_equals_pi_l2904_290416

theorem square_side_equals_pi :
  ∀ x : ℝ,
  (4 * x = 2 * π * 2) →
  x = π :=
by
  sorry

end square_side_equals_pi_l2904_290416


namespace intersection_equality_implies_a_range_l2904_290408

def A : Set ℝ := {x | (1/2 : ℝ) ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem intersection_equality_implies_a_range (a : ℝ) :
  (Aᶜ ∩ B a = B a) → a ≥ -1/4 := by
  sorry

end intersection_equality_implies_a_range_l2904_290408


namespace complex_inequality_l2904_290467

theorem complex_inequality (z₁ z₂ z₃ z₄ : ℂ) :
  Complex.abs (z₁ - z₃)^2 + Complex.abs (z₂ - z₄)^2 ≤
  Complex.abs (z₁ - z₂)^2 + Complex.abs (z₂ - z₃)^2 +
  Complex.abs (z₃ - z₄)^2 + Complex.abs (z₄ - z₁)^2 ∧
  (Complex.abs (z₁ - z₃)^2 + Complex.abs (z₂ - z₄)^2 =
   Complex.abs (z₁ - z₂)^2 + Complex.abs (z₂ - z₃)^2 +
   Complex.abs (z₃ - z₄)^2 + Complex.abs (z₄ - z₁)^2 ↔
   z₁ + z₃ = z₂ + z₄) := by
  sorry

end complex_inequality_l2904_290467


namespace smallest_four_digit_solution_l2904_290413

theorem smallest_four_digit_solution (x : ℕ) : x = 1053 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧ 
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (9 * y ≡ 27 [ZMOD 15] ∧
     3 * y + 15 ≡ 21 [ZMOD 8] ∧
     -3 * y + 4 ≡ 2 * y + 5 [ZMOD 16]) →
    x ≤ y) ∧
  (9 * x ≡ 27 [ZMOD 15]) ∧
  (3 * x + 15 ≡ 21 [ZMOD 8]) ∧
  (-3 * x + 4 ≡ 2 * x + 5 [ZMOD 16]) :=
by sorry

end smallest_four_digit_solution_l2904_290413


namespace f_even_h_odd_l2904_290453

-- Define the functions f and h
def f (x : ℝ) : ℝ := x^2
def h (x : ℝ) : ℝ := x

-- State the theorem
theorem f_even_h_odd : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, h (-x) = -h x) := by
  sorry

end f_even_h_odd_l2904_290453


namespace data_transmission_time_l2904_290404

theorem data_transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 100 →
  chunks_per_block = 800 →
  transmission_rate = 200 →
  (blocks * chunks_per_block : ℝ) / transmission_rate / 60 = 6.666666666666667 :=
by sorry

end data_transmission_time_l2904_290404


namespace math_club_team_selection_l2904_290471

def math_club_selection (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (boys_in_team : ℕ) (girls_in_team : ℕ) : ℕ :=
  (total_boys.choose boys_in_team) * (total_girls.choose girls_in_team)

theorem math_club_team_selection :
  math_club_selection 10 12 8 4 4 = 103950 := by
sorry

end math_club_team_selection_l2904_290471
