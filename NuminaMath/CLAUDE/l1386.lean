import Mathlib

namespace simple_interest_problem_l1386_138658

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  interest = 4016.25 →
  rate = 0.01 →
  time = 5 →
  interest = principal * rate * time →
  principal = 80325 :=
by sorry

end simple_interest_problem_l1386_138658


namespace pages_left_to_read_l1386_138694

/-- Given a book with 400 pages, prove that if a person has read 20% of it,
    they need to read 320 pages to finish the book. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) 
  (h1 : total_pages = 400)
  (h2 : percentage_read = 20 / 100) :
  total_pages - (total_pages * percentage_read).floor = 320 := by
  sorry

end pages_left_to_read_l1386_138694


namespace students_taking_neither_music_nor_art_l1386_138666

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500) 
  (h2 : music_students = 20) 
  (h3 : art_students = 20) 
  (h4 : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 470 := by
  sorry

#check students_taking_neither_music_nor_art

end students_taking_neither_music_nor_art_l1386_138666


namespace fair_attendance_l1386_138626

/-- Represents the number of children attending the fair -/
def num_children : ℕ := sorry

/-- Represents the number of adults attending the fair -/
def num_adults : ℕ := sorry

/-- The admission fee for children in cents -/
def child_fee : ℕ := 150

/-- The admission fee for adults in cents -/
def adult_fee : ℕ := 400

/-- The total number of people attending the fair -/
def total_people : ℕ := 2200

/-- The total amount collected in cents -/
def total_amount : ℕ := 505000

theorem fair_attendance : 
  num_children + num_adults = total_people ∧
  num_children * child_fee + num_adults * adult_fee = total_amount →
  num_children = 1500 :=
sorry

end fair_attendance_l1386_138626


namespace smallest_other_integer_l1386_138686

theorem smallest_other_integer (a b x : ℕ+) : 
  (a = 36 ∨ b = 36) →
  Nat.gcd a b = x + 6 →
  Nat.lcm a b = x * (x + 6) →
  (a ≠ 36 → a ≥ 24) ∧ (b ≠ 36 → b ≥ 24) :=
sorry

end smallest_other_integer_l1386_138686


namespace binomial_inequality_l1386_138687

theorem binomial_inequality (n : ℕ) : 2 ≤ (1 + 1 / n : ℝ) ^ n ∧ (1 + 1 / n : ℝ) ^ n < 3 := by
  sorry

end binomial_inequality_l1386_138687


namespace fifteenth_term_equals_44_l1386_138674

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- The 15th term of the specific arithmetic progression -/
def fifteenthTerm : ℝ :=
  arithmeticProgressionTerm 2 3 15

theorem fifteenth_term_equals_44 : fifteenthTerm = 44 := by
  sorry

end fifteenth_term_equals_44_l1386_138674


namespace red_balls_count_l1386_138613

/-- The number of red balls in a bag with given conditions -/
theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) 
    (prob_not_red_purple : ℚ) (h1 : total = 100) (h2 : white = 50) (h3 : green = 30) 
    (h4 : yellow = 10) (h5 : purple = 3) (h6 : prob_not_red_purple = 9/10) : 
    total - (white + green + yellow + purple) = 7 := by
  sorry

end red_balls_count_l1386_138613


namespace parabola_focus_directrix_distance_l1386_138621

/-- For a parabola given by the equation y^2 = 10x, the distance from its focus to its directrix is 5. -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = 10*x → (∃ (focus_x focus_y directrix_x : ℝ),
    (∀ (point_x point_y : ℝ), point_y^2 = 10*point_x ↔ 
      (point_x - focus_x)^2 + (point_y - focus_y)^2 = (point_x - directrix_x)^2) ∧
    |focus_x - directrix_x| = 5) :=
by sorry

end parabola_focus_directrix_distance_l1386_138621


namespace inequality_proof_l1386_138622

theorem inequality_proof (b : ℝ) : (3*b - 1)*(4*b + 1) > (2*b + 1)*(5*b - 3) := by
  sorry

end inequality_proof_l1386_138622


namespace negation_of_existence_negation_of_exponential_equation_l1386_138680

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by
  sorry

theorem negation_of_exponential_equation : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) ↔ (∀ x : ℝ, Real.exp x ≠ x - 1) := by
  sorry

end negation_of_existence_negation_of_exponential_equation_l1386_138680


namespace pure_imaginary_complex_fraction_l1386_138659

/-- If z = (a + i) / (1 - i) is a pure imaginary number and a is real, then a = 1 -/
theorem pure_imaginary_complex_fraction (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ b : ℝ, z = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_complex_fraction_l1386_138659


namespace equation_solution_l1386_138643

theorem equation_solution :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0 :=
by
  sorry

end equation_solution_l1386_138643


namespace intersection_point_d_l1386_138650

def g (c : ℤ) (x : ℝ) : ℝ := 5 * x + c

theorem intersection_point_d (c : ℤ) (d : ℤ) :
  (g c (-5) = d) ∧ (g c d = -5) → d = -5 := by sorry

end intersection_point_d_l1386_138650


namespace cone_volume_l1386_138675

-- Define the right triangle
structure RightTriangle where
  area : ℝ
  centroidCircumference : ℝ

-- Define the cone formed by rotating the right triangle
structure Cone where
  triangle : RightTriangle

-- Define the volume of the cone
def volume (c : Cone) : ℝ := c.triangle.area * c.triangle.centroidCircumference

-- Theorem statement
theorem cone_volume (c : Cone) : volume c = c.triangle.area * c.triangle.centroidCircumference := by
  sorry

end cone_volume_l1386_138675


namespace vertices_must_be_even_l1386_138624

-- Define a polyhedron
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

-- Define a property for trihedral angles
def has_trihedral_angles (p : Polyhedron) : Prop :=
  3 * p.vertices = 2 * p.edges

-- Theorem statement
theorem vertices_must_be_even (p : Polyhedron) 
  (h : has_trihedral_angles p) : Even p.vertices := by
  sorry


end vertices_must_be_even_l1386_138624


namespace lefty_points_lefty_scored_20_points_l1386_138645

theorem lefty_points : ℝ → Prop :=
  fun L : ℝ =>
    let righty : ℝ := L / 2
    let third_teammate : ℝ := 3 * L
    let total_points : ℝ := L + righty + third_teammate
    let average_points : ℝ := total_points / 3
    average_points = 30 → L = 20

-- Proof
theorem lefty_scored_20_points : ∃ L : ℝ, lefty_points L :=
  sorry

end lefty_points_lefty_scored_20_points_l1386_138645


namespace derivative_f_at_zero_l1386_138679

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.tan (x^3 + x^2 * Real.sin (2/x))
  else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end derivative_f_at_zero_l1386_138679


namespace parking_capacity_l1386_138631

/-- Represents a parking garage with four levels --/
structure ParkingGarage :=
  (level1 : ℕ)
  (level2 : ℕ)
  (level3 : ℕ)
  (level4 : ℕ)

/-- Calculates the total number of parking spaces in the garage --/
def total_spaces (g : ParkingGarage) : ℕ :=
  g.level1 + g.level2 + g.level3 + g.level4

/-- Theorem: Given the parking garage conditions, 299 more cars can be accommodated --/
theorem parking_capacity 
  (g : ParkingGarage)
  (h1 : g.level1 = 90)
  (h2 : g.level2 = g.level1 + 8)
  (h3 : g.level3 = g.level2 + 12)
  (h4 : g.level4 = g.level3 - 9)
  (h5 : total_spaces g - 100 = 299) : 
  ∃ (n : ℕ), n = 299 ∧ n = total_spaces g - 100 :=
by
  sorry

#check parking_capacity

end parking_capacity_l1386_138631


namespace election_winner_percentage_l1386_138607

theorem election_winner_percentage : 
  let votes : List ℕ := [1036, 4636, 11628]
  let total_votes := votes.sum
  let winning_votes := votes.maximum?
  let winning_percentage := (winning_votes.getD 0 : ℚ) / total_votes * 100
  winning_percentage = 67.2 := by
sorry

end election_winner_percentage_l1386_138607


namespace banana_boxes_l1386_138649

def total_bananas : ℕ := 40
def bananas_per_box : ℕ := 4

theorem banana_boxes : total_bananas / bananas_per_box = 10 := by
  sorry

end banana_boxes_l1386_138649


namespace ones_divisible_by_l_l1386_138628

theorem ones_divisible_by_l (l : ℕ) (h1 : ¬ 2 ∣ l) (h2 : ¬ 5 ∣ l) :
  ∃ n : ℕ, l ∣ n ∧ ∀ d : ℕ, d ∈ (n.digits 10) → d = 1 :=
sorry

end ones_divisible_by_l_l1386_138628


namespace first_solution_carbonated_water_percentage_l1386_138608

/-- Represents a solution with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_100 : lemonade + carbonated_water = 100

/-- Proves that the first solution is 80% carbonated water given the conditions -/
theorem first_solution_carbonated_water_percentage
  (solution1 : Solution)
  (solution2 : Solution)
  (h1 : solution1.lemonade = 20)
  (h2 : solution2.lemonade = 45)
  (h3 : solution2.carbonated_water = 55)
  (h_mixture : 0.5 * solution1.carbonated_water + 0.5 * solution2.carbonated_water = 67.5) :
  solution1.carbonated_water = 80 := by
  sorry

#check first_solution_carbonated_water_percentage

end first_solution_carbonated_water_percentage_l1386_138608


namespace corrected_mean_calculation_l1386_138646

def original_mean : ℝ := 36
def num_observations : ℕ := 50
def error_1 : (ℝ × ℝ) := (46, 23)
def error_2 : (ℝ × ℝ) := (55, 40)
def error_3 : (ℝ × ℝ) := (28, 15)

theorem corrected_mean_calculation :
  let total_sum := original_mean * num_observations
  let error_sum := error_1.1 + error_2.1 + error_3.1 - (error_1.2 + error_2.2 + error_3.2)
  let corrected_sum := total_sum + error_sum
  corrected_sum / num_observations = 37.02 := by sorry

end corrected_mean_calculation_l1386_138646


namespace imaginary_part_of_z_squared_l1386_138614

theorem imaginary_part_of_z_squared (z : ℂ) (h : z * (1 - Complex.I) = 2) : 
  Complex.im (z^2) = 2 :=
sorry

end imaginary_part_of_z_squared_l1386_138614


namespace inverse_proportion_through_neg_one_three_l1386_138685

/-- An inverse proportion function passing through (-1, 3) has k = -3 --/
theorem inverse_proportion_through_neg_one_three (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (k / x = 3 ↔ x = -1)) → k = -3 := by
  sorry

end inverse_proportion_through_neg_one_three_l1386_138685


namespace distance_origin_to_line_l1386_138639

/-- The distance from the origin to the line x + 2y - 5 = 0 is √5 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + 2*y - 5 = 0}
  abs (5) / Real.sqrt (1^2 + 2^2) = Real.sqrt 5 := by
sorry

end distance_origin_to_line_l1386_138639


namespace min_sum_grid_l1386_138671

theorem min_sum_grid (a b c d : ℕ+) (h : a * b + c * d + a * c + b * d = 2015) :
  a + b + c + d ≥ 88 := by
  sorry

end min_sum_grid_l1386_138671


namespace permutation_combination_problem_l1386_138696

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem permutation_combination_problem :
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 + A 9 5) = 5 / 11 ∧
  C 200 192 + C 200 196 + 2 * C 200 197 = 67331650 := by
  sorry

end permutation_combination_problem_l1386_138696


namespace photocopy_cost_calculation_l1386_138637

/-- The cost of a single photocopy --/
def photocopy_cost : ℝ := sorry

/-- The discount rate for orders over 100 photocopies --/
def discount_rate : ℝ := 0.25

/-- The number of copies each person needs --/
def copies_per_person : ℕ := 80

/-- The amount saved per person when ordering together --/
def savings_per_person : ℝ := 0.40

/-- Theorem stating the cost of a single photocopy --/
theorem photocopy_cost_calculation : photocopy_cost = 0.02 := by
  have h1 : 2 * copies_per_person * photocopy_cost - 
    (2 * copies_per_person * photocopy_cost * (1 - discount_rate)) = 
    2 * savings_per_person := by sorry
  
  -- The rest of the proof steps would go here
  sorry

end photocopy_cost_calculation_l1386_138637


namespace line_parallel_to_intersection_l1386_138635

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and intersection relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- State the theorem
theorem line_parallel_to_intersection
  (m n : Line)
  (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_intersection : intersection α β n)
  (h_m_parallel_α : parallel_plane m α)
  (h_m_parallel_β : parallel_plane m β) :
  parallel m n :=
sorry

end line_parallel_to_intersection_l1386_138635


namespace cubic_equation_implies_specific_value_l1386_138693

theorem cubic_equation_implies_specific_value :
  ∀ x : ℝ, x^3 - 3 * Real.sqrt 2 * x^2 + 6 * x - 2 * Real.sqrt 2 - 8 = 0 →
  x^5 - 41 * x^2 + 2012 = 1998 := by
  sorry

end cubic_equation_implies_specific_value_l1386_138693


namespace matrix_multiplication_result_l1386_138655

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -3; 0, 3, -1; -1, 3, 2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, -1, 0; 2, 1, -2; 3, 0, 1]
def c : ℤ := 2

theorem matrix_multiplication_result :
  c • (A * B) = !![(-14:ℤ), -4, -6; 6, 6, -14; 22, 8, -8] := by sorry

end matrix_multiplication_result_l1386_138655


namespace complex_fraction_equality_l1386_138610

theorem complex_fraction_equality : ∀ (i : ℂ), i^2 = -1 → (5*i)/(1+2*i) = 2+i := by
  sorry

end complex_fraction_equality_l1386_138610


namespace specific_enclosed_area_l1386_138632

/-- The area enclosed by a curve composed of 9 congruent circular arcs, where the centers of the
    corresponding circles are among the vertices of a regular hexagon. -/
def enclosed_area (arc_length : ℝ) (hexagon_side : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the area enclosed by the specific curve described in the problem
    is equal to (27√3)/2 + (1125π²)/96. -/
theorem specific_enclosed_area :
  enclosed_area (5 * π / 6) 3 = (27 * Real.sqrt 3) / 2 + (1125 * π^2) / 96 := by
  sorry

end specific_enclosed_area_l1386_138632


namespace pencil_distribution_l1386_138629

/-- Given a total number of pencils and students, calculate the number of pencils per student -/
def pencils_per_student (total_pencils : ℕ) (total_students : ℕ) : ℕ :=
  total_pencils / total_students

theorem pencil_distribution (total_pencils : ℕ) (total_students : ℕ) 
  (h1 : total_pencils = 195)
  (h2 : total_students = 65) :
  pencils_per_student total_pencils total_students = 3 := by
  sorry

end pencil_distribution_l1386_138629


namespace local_extremum_implies_b_minus_a_l1386_138661

/-- A function with a local extremum -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

theorem local_extremum_implies_b_minus_a (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → b - a = 15 := by
  sorry

end local_extremum_implies_b_minus_a_l1386_138661


namespace total_tips_calculation_l1386_138617

def lawn_price : ℕ := 33
def lawns_mowed : ℕ := 16
def total_earned : ℕ := 558

theorem total_tips_calculation : 
  total_earned - (lawn_price * lawns_mowed) = 30 := by sorry

end total_tips_calculation_l1386_138617


namespace correct_num_selections_l1386_138657

/-- The number of pairs of gloves -/
def num_pairs : ℕ := 5

/-- The number of gloves to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select 3 gloves of different colors from 5 pairs of gloves -/
def num_selections : ℕ := 80

/-- Theorem stating that the number of selections is correct -/
theorem correct_num_selections :
  (num_pairs.choose num_selected) * (2^num_selected) = num_selections :=
by sorry

end correct_num_selections_l1386_138657


namespace octal_127_equals_87_l1386_138676

-- Define the octal number as a list of digits
def octal_127 : List Nat := [1, 2, 7]

-- Function to convert octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_127_equals_87 :
  octal_to_decimal octal_127 = 87 := by
  sorry

end octal_127_equals_87_l1386_138676


namespace divisible_by_4_or_6_count_l1386_138691

def count_divisible (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

theorem divisible_by_4_or_6_count :
  (count_divisible 51 4) + (count_divisible 51 6) - (count_divisible 51 12) = 16 := by
  sorry

end divisible_by_4_or_6_count_l1386_138691


namespace hyperbola_condition_ellipse_x_major_condition_l1386_138656

-- Define the curve C
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) := ∀ (x y : ℝ), (x, y) ∈ C t → (4 - t) * (t - 1) < 0

-- Define what it means for C to be an ellipse with major axis on x-axis
def is_ellipse_x_major (t : ℝ) := ∀ (x y : ℝ), (x, y) ∈ C t → (4 - t) > (t - 1) ∧ (t - 1) > 0

-- Theorem statements
theorem hyperbola_condition (t : ℝ) :
  is_hyperbola t → t > 4 ∨ t < 1 := by sorry

theorem ellipse_x_major_condition (t : ℝ) :
  is_ellipse_x_major t → 1 < t ∧ t < 5/2 := by sorry

end hyperbola_condition_ellipse_x_major_condition_l1386_138656


namespace french_exam_vocabulary_l1386_138641

theorem french_exam_vocabulary (total_words : ℕ) (guess_rate : ℚ) (target_score : ℚ) : 
  total_words = 600 → 
  guess_rate = 5 / 100 → 
  target_score = 90 / 100 → 
  ∃ (words_to_learn : ℕ), 
    words_to_learn ≥ 537 ∧ 
    (words_to_learn : ℚ) / total_words + 
      guess_rate * ((total_words - words_to_learn) : ℚ) / total_words ≥ target_score ∧
    ∀ (x : ℕ), x < 537 → 
      (x : ℚ) / total_words + 
        guess_rate * ((total_words - x) : ℚ) / total_words < target_score :=
by sorry

end french_exam_vocabulary_l1386_138641


namespace angstadt_seniors_l1386_138603

/-- Mr. Angstadt's class enrollment problem -/
theorem angstadt_seniors (total_students : ℕ) 
  (stats_percent geometry_percent : ℚ)
  (stats_calc_overlap geometry_calc_overlap : ℚ)
  (stats_senior_percent geometry_senior_percent calc_senior_percent : ℚ)
  (h1 : total_students = 240)
  (h2 : stats_percent = 45/100)
  (h3 : geometry_percent = 35/100)
  (h4 : stats_calc_overlap = 10/100)
  (h5 : geometry_calc_overlap = 5/100)
  (h6 : stats_senior_percent = 90/100)
  (h7 : geometry_senior_percent = 60/100)
  (h8 : calc_senior_percent = 80/100) :
  ∃ (senior_count : ℕ), senior_count = 161 := by
sorry


end angstadt_seniors_l1386_138603


namespace vertical_angles_are_equal_equal_angles_are_vertical_converse_of_vertical_angles_are_equal_l1386_138620

/-- Definition of vertical angles -/
def VerticalAngles (α β : Angle) : Prop := sorry

/-- The original proposition -/
theorem vertical_angles_are_equal (α β : Angle) : 
  VerticalAngles α β → α = β := sorry

/-- The converse proposition -/
theorem equal_angles_are_vertical (α β : Angle) : 
  α = β → VerticalAngles α β := sorry

/-- Theorem stating that the converse of "Vertical angles are equal" 
    is "Angles that are equal are vertical angles" -/
theorem converse_of_vertical_angles_are_equal :
  (∀ α β : Angle, VerticalAngles α β → α = β) ↔ 
  (∀ α β : Angle, α = β → VerticalAngles α β) :=
sorry

end vertical_angles_are_equal_equal_angles_are_vertical_converse_of_vertical_angles_are_equal_l1386_138620


namespace loaves_needed_l1386_138634

/-- The number of first-year students -/
def first_year_students : ℕ := 247

/-- The difference between the number of sophomores and first-year students -/
def sophomore_difference : ℕ := 131

/-- The number of sophomores -/
def sophomores : ℕ := first_year_students + sophomore_difference

/-- The total number of students (first-year and sophomores) -/
def total_students : ℕ := first_year_students + sophomores

theorem loaves_needed : total_students = 625 := by sorry

end loaves_needed_l1386_138634


namespace special_circle_equation_l1386_138673

/-- A circle with center on the y-axis passing through (3, 1) and tangent to x-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  passes_through_point : (3 - center.1)^2 + (1 - center.2)^2 = radius^2
  tangent_to_x_axis : center.2 = radius

/-- The equation of the special circle is x^2 + y^2 - 10y = 0 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ x^2 + y^2 - 10*y = 0 :=
by sorry

end special_circle_equation_l1386_138673


namespace remainder_divisibility_l1386_138612

theorem remainder_divisibility (N : ℤ) : 
  (N % 779 = 47) → (N % 19 = 9) := by
sorry

end remainder_divisibility_l1386_138612


namespace fair_cake_distribution_l1386_138690

/-- Represents a cake flavor -/
inductive Flavor
  | Chocolate
  | Strawberry
  | Vanilla

/-- Represents a child's flavor preferences -/
structure ChildPreference where
  flavor1 : Flavor
  flavor2 : Flavor
  different : flavor1 ≠ flavor2

/-- Represents the distribution of cakes -/
structure CakeDistribution where
  totalCakes : Nat
  numChildren : Nat
  numFlavors : Nat
  childPreferences : Fin numChildren → ChildPreference
  cakesPerChild : Nat
  cakesPerFlavor : Fin numFlavors → Nat

/-- Theorem stating that a fair distribution is possible -/
theorem fair_cake_distribution 
  (d : CakeDistribution) 
  (h_total : d.totalCakes = 18) 
  (h_children : d.numChildren = 3) 
  (h_flavors : d.numFlavors = 3) 
  (h_preferences : ∀ i, (d.childPreferences i).flavor1 ≠ (d.childPreferences i).flavor2) 
  (h_distribution : ∀ i, d.cakesPerFlavor i = 6) :
  d.cakesPerChild = 6 ∧ 
  (∀ i : Fin d.numChildren, ∃ f1 f2 : Fin d.numFlavors, 
    f1 ≠ f2 ∧ 
    d.cakesPerFlavor f1 + d.cakesPerFlavor f2 = d.cakesPerChild) :=
by sorry

end fair_cake_distribution_l1386_138690


namespace probability_hit_at_least_once_l1386_138689

-- Define the probability of hitting the target in a single shot
def hit_probability : ℚ := 2/3

-- Define the number of shots
def num_shots : ℕ := 3

-- Theorem statement
theorem probability_hit_at_least_once :
  1 - (1 - hit_probability) ^ num_shots = 26/27 := by
  sorry

end probability_hit_at_least_once_l1386_138689


namespace imaginary_part_of_i_power_2017_l1386_138633

theorem imaginary_part_of_i_power_2017 : Complex.im (Complex.I ^ 2017) = 1 := by
  sorry

end imaginary_part_of_i_power_2017_l1386_138633


namespace complement_of_intersection_l1386_138616

def U : Set ℕ := {1,2,3,4,5}
def M : Set ℕ := {1,2,4}
def N : Set ℕ := {3,4,5}

theorem complement_of_intersection :
  (M ∩ N)ᶜ = {1,2,3,5} :=
by sorry

end complement_of_intersection_l1386_138616


namespace jack_baseball_cards_l1386_138681

theorem jack_baseball_cards :
  ∀ (total_cards baseball_cards football_cards : ℕ),
  total_cards = 125 →
  baseball_cards = 3 * football_cards + 5 →
  total_cards = baseball_cards + football_cards →
  baseball_cards = 95 := by
sorry

end jack_baseball_cards_l1386_138681


namespace base_conversion_subtraction_l1386_138618

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5 + c

theorem base_conversion_subtraction :
  base6ToBase10 3 5 4 - base5ToBase10 2 3 1 = 76 := by sorry

end base_conversion_subtraction_l1386_138618


namespace intersection_complement_theorem_l1386_138627

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 4}
def B : Set Nat := {4, 5}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {2} := by sorry

end intersection_complement_theorem_l1386_138627


namespace consecutive_integers_around_sqrt3_l1386_138697

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end consecutive_integers_around_sqrt3_l1386_138697


namespace multiplication_mistake_l1386_138604

theorem multiplication_mistake (x : ℕ) : x = 43 := by
  have h1 : 136 * x - 1224 = 136 * 34 := by sorry
  sorry

end multiplication_mistake_l1386_138604


namespace coordinate_sum_of_h_l1386_138601

/-- Given a function g where g(3) = 8, and a function h where h(x) = (g(x))^3 for all x,
    the sum of the coordinates of the point (3, h(3)) is 515. -/
theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) 
    (hg : g 3 = 8) (hh : ∀ x, h x = (g x)^3) : 
    3 + h 3 = 515 := by
  sorry

end coordinate_sum_of_h_l1386_138601


namespace average_height_is_10_8_l1386_138667

def tree_heights (h1 h2 h3 h4 h5 : ℕ) : Prop :=
  h2 = 18 ∧
  (h1 = 3 * h2 ∨ h1 * 3 = h2) ∧
  (h2 = 3 * h3 ∨ h2 * 3 = h3) ∧
  (h3 = 3 * h4 ∨ h3 * 3 = h4) ∧
  (h4 = 3 * h5 ∨ h4 * 3 = h5)

theorem average_height_is_10_8 :
  ∃ (h1 h2 h3 h4 h5 : ℕ), tree_heights h1 h2 h3 h4 h5 ∧
  (h1 + h2 + h3 + h4 + h5) / 5 = 54 / 5 :=
sorry

end average_height_is_10_8_l1386_138667


namespace range_of_a_for_p_and_q_l1386_138682

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + a*x

def is_monotonically_increasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

def represents_hyperbola (a : ℝ) : Prop :=
  (a + 2) * (a - 2) < 0

theorem range_of_a_for_p_and_q :
  {a : ℝ | is_monotonically_increasing (f a) ∧ represents_hyperbola a} = Set.Icc 0 2 :=
sorry

end range_of_a_for_p_and_q_l1386_138682


namespace equation_has_root_minus_one_l1386_138665

theorem equation_has_root_minus_one : ∃ x : ℝ, x = -1 ∧ x^2 - x - 2 = 0 := by
  sorry

end equation_has_root_minus_one_l1386_138665


namespace shaded_area_is_one_third_l1386_138600

/-- Represents a 3x3 square quilt block -/
structure QuiltBlock :=
  (size : Nat)
  (shaded_area : ℚ)

/-- The size of the quilt block is 3 -/
def quilt_size : Nat := 3

/-- The quilt block with the given shaded pattern -/
def patterned_quilt : QuiltBlock :=
  { size := quilt_size,
    shaded_area := 1 }

/-- Theorem stating that the shaded area of the patterned quilt is 1/3 of the total area -/
theorem shaded_area_is_one_third (q : QuiltBlock) (h : q = patterned_quilt) :
  q.shaded_area / (q.size * q.size : ℚ) = 1 / 3 := by
  sorry

end shaded_area_is_one_third_l1386_138600


namespace cubic_roots_nature_l1386_138642

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4

-- Theorem statement
theorem cubic_roots_nature :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c < 0 ∧
    (∀ x : ℝ, cubic_poly x = 0 ↔ (x = a ∨ x = b ∨ x = c))) :=
sorry

end cubic_roots_nature_l1386_138642


namespace marble_probability_l1386_138647

theorem marble_probability (blue green white : ℝ) 
  (prob_sum : blue + green + white = 1)
  (prob_blue : blue = 0.25)
  (prob_green : green = 0.4) :
  white = 0.35 := by
  sorry

end marble_probability_l1386_138647


namespace x_value_l1386_138652

theorem x_value (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 := by
  sorry

end x_value_l1386_138652


namespace music_spending_l1386_138625

theorem music_spending (total_allowance : ℝ) (music_fraction : ℝ) : 
  total_allowance = 50 → music_fraction = 3/10 → music_fraction * total_allowance = 15 := by
  sorry

end music_spending_l1386_138625


namespace quadratic_trinomial_minimum_l1386_138688

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  ∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ 
    (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ m) ∧
    (∃ x : ℝ, (a^2 + b^2) / (a - b) = m) := by
  sorry

end quadratic_trinomial_minimum_l1386_138688


namespace solution_satisfies_equation_is_linear_in_two_variables_l1386_138699

-- Define the solution point
def solution_x : ℝ := 2
def solution_y : ℝ := -3

-- Define the linear equation
def linear_equation (x y : ℝ) : Prop := x + y = -1

-- Theorem statement
theorem solution_satisfies_equation : 
  linear_equation solution_x solution_y := by
  sorry

-- Theorem to prove the equation is linear in two variables
theorem is_linear_in_two_variables : 
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
  ∀ (x y : ℝ), linear_equation x y ↔ a * x + b * y + c = 0 := by
  sorry

end solution_satisfies_equation_is_linear_in_two_variables_l1386_138699


namespace fraction_sum_l1386_138654

theorem fraction_sum (p q : ℝ) (h : p ≠ 0 ∧ q ≠ 0) 
  (h1 : 1/p + 1/q = 1/(p+q)) : p/q + q/p = -1 := by
  sorry

end fraction_sum_l1386_138654


namespace oranges_left_to_sell_l1386_138683

theorem oranges_left_to_sell (x : ℕ) (h : x ≥ 7) :
  let total := 12 * x
  let friend1 := (1 / 4 : ℚ) * total
  let friend2 := (1 / 6 : ℚ) * total
  let charity := (1 / 8 : ℚ) * total
  let remaining_after_giving := total - friend1 - friend2 - charity
  let sold_yesterday := (3 / 7 : ℚ) * remaining_after_giving
  let remaining_after_selling := remaining_after_giving - sold_yesterday
  let eaten_by_birds := (1 / 10 : ℚ) * remaining_after_selling
  let remaining_after_birds := remaining_after_selling - eaten_by_birds
  remaining_after_birds - 4 = (3.0214287 : ℚ) * x - 4 := by sorry

end oranges_left_to_sell_l1386_138683


namespace intersection_equality_implies_range_l1386_138669

-- Define the sets A and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | -a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem intersection_equality_implies_range (a : ℝ) :
  C a ∩ A = C a → -3/2 ≤ a ∧ a ≤ -1 :=
by sorry

end intersection_equality_implies_range_l1386_138669


namespace edward_lives_problem_l1386_138668

theorem edward_lives_problem (lives_lost lives_remaining : ℕ) 
  (h1 : lives_lost = 8)
  (h2 : lives_remaining = 7) :
  lives_lost + lives_remaining = 15 :=
by sorry

end edward_lives_problem_l1386_138668


namespace largest_house_number_l1386_138638

def phone_number : List Nat := [3, 4, 6, 2, 8, 9, 0]

def sum_digits (num : List Nat) : Nat :=
  num.foldl (· + ·) 0

def is_distinct (num : List Nat) : Prop :=
  num.length = num.toFinset.card

def is_valid_house_number (num : List Nat) : Prop :=
  num.length = 4 ∧ is_distinct num ∧ sum_digits num = sum_digits phone_number

theorem largest_house_number :
  ∀ (house_num : List Nat),
    is_valid_house_number house_num →
    house_num.foldl (fun acc d => acc * 10 + d) 0 ≤ 9876 :=
by sorry

end largest_house_number_l1386_138638


namespace car_journey_speed_l1386_138636

theorem car_journey_speed (s : ℝ) (h : s > 0) : 
  let first_part := 0.4 * s
  let second_part := 0.6 * s
  let first_speed := 40
  let average_speed := 100
  let first_time := first_part / first_speed
  let total_time := s / average_speed
  ∃ d : ℝ, d > 0 ∧ second_part / d = total_time - first_time ∧ d = 120 := by
sorry

end car_journey_speed_l1386_138636


namespace walnut_trees_after_planting_l1386_138684

/-- The number of walnut trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of trees planted. -/
theorem walnut_trees_after_planting 
  (initial_trees : ℕ) 
  (planted_trees : ℕ) : 
  initial_trees + planted_trees = 
    initial_trees + planted_trees :=
by sorry

/-- Specific instance of the theorem with 4 initial trees and 6 planted trees -/
example : 4 + 6 = 10 :=
by sorry

end walnut_trees_after_planting_l1386_138684


namespace count_propositions_is_two_l1386_138623

-- Define a type for statements
inductive Statement
| EmptySetProperSubset
| QuadraticInequality
| PerpendicularLinesQuestion
| NaturalNumberEven

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Bool :=
  match s with
  | Statement.EmptySetProperSubset => true
  | Statement.QuadraticInequality => false
  | Statement.PerpendicularLinesQuestion => false
  | Statement.NaturalNumberEven => true

-- Define a function to count propositions
def countPropositions (statements : List Statement) : Nat :=
  statements.filter isProposition |>.length

-- Theorem to prove
theorem count_propositions_is_two :
  let statements := [Statement.EmptySetProperSubset, Statement.QuadraticInequality,
                     Statement.PerpendicularLinesQuestion, Statement.NaturalNumberEven]
  countPropositions statements = 2 := by
  sorry

end count_propositions_is_two_l1386_138623


namespace min_tiles_for_coverage_l1386_138660

-- Define the grid size
def grid_size : ℕ := 8

-- Define the size of small squares
def small_square_size : ℕ := 2

-- Define the number of cells covered by each L-shaped tile
def cells_per_tile : ℕ := 3

-- Calculate the number of small squares in the grid
def num_small_squares : ℕ := (grid_size * grid_size) / (small_square_size * small_square_size)

-- Define the minimum number of cells that need to be covered
def min_cells_to_cover : ℕ := 2 * num_small_squares

-- Define the minimum number of L-shaped tiles needed
def min_tiles_needed : ℕ := (min_cells_to_cover + cells_per_tile - 1) / cells_per_tile

-- Theorem statement
theorem min_tiles_for_coverage : min_tiles_needed = 11 := by
  sorry

end min_tiles_for_coverage_l1386_138660


namespace expansion_equality_l1386_138677

theorem expansion_equality (x : ℝ) : (1 + x^2) * (1 - x^4) = 1 + x^2 - x^4 - x^6 := by
  sorry

end expansion_equality_l1386_138677


namespace rational_solutions_k_l1386_138692

/-- A function that checks if a given positive integer k results in rational solutions for the equation kx^2 + 20x + k = 0 -/
def has_rational_solutions (k : ℕ+) : Prop :=
  ∃ n : ℕ, (100 - k.val^2 : ℤ) = n^2

/-- The theorem stating that the positive integer values of k for which kx^2 + 20x + k = 0 has rational solutions are exactly 6, 8, and 10 -/
theorem rational_solutions_k :
  ∀ k : ℕ+, has_rational_solutions k ↔ k.val ∈ ({6, 8, 10} : Set ℕ) := by sorry

end rational_solutions_k_l1386_138692


namespace bonus_sector_area_l1386_138644

/-- Given a circular spinner with radius 15 cm and a "Bonus" sector with a 
    probability of 1/3 of being landed on, the area of the "Bonus" sector 
    is 75π square centimeters. -/
theorem bonus_sector_area (radius : ℝ) (probability : ℝ) (bonus_area : ℝ) : 
  radius = 15 →
  probability = 1 / 3 →
  bonus_area = probability * π * radius^2 →
  bonus_area = 75 * π := by
  sorry


end bonus_sector_area_l1386_138644


namespace ratio_problem_l1386_138640

theorem ratio_problem (x y : ℝ) (h : (2*x - y) / (x + y) = 2/3) : x / y = 5/4 := by
  sorry

end ratio_problem_l1386_138640


namespace horner_v4_value_l1386_138630

def horner_step (v : ℤ) (a : ℤ) (x : ℤ) : ℤ := v * x + a

def horner_method (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc coeff => horner_step acc coeff x) 0

theorem horner_v4_value :
  let coeffs := [3, 5, 6, 20, -8, 35, 12]
  let x := -2
  let v0 := 3
  let v1 := horner_step v0 5 x
  let v2 := horner_step v1 6 x
  let v3 := horner_step v2 20 x
  let v4 := horner_step v3 (-8) x
  v4 = -16 := by sorry

end horner_v4_value_l1386_138630


namespace quarters_to_nickels_difference_l1386_138662

/-- The difference in money (in nickels) between two people with different numbers of quarters -/
theorem quarters_to_nickels_difference (q : ℚ) : 
  5 * ((7 * q + 2) - (3 * q + 7)) = 20 * (q - 1.25) := by
  sorry

end quarters_to_nickels_difference_l1386_138662


namespace diophantine_equation_unique_solution_l1386_138615

theorem diophantine_equation_unique_solution :
  ∀ x y z t : ℤ, x^2 + y^2 + z^2 + t^2 = 2*x*y*z*t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end diophantine_equation_unique_solution_l1386_138615


namespace min_gumballs_for_four_same_is_thirteen_l1386_138611

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSame (machine : GumballMachine) : ℕ := 13

/-- Theorem stating that for the given gumball machine configuration, 
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_is_thirteen (machine : GumballMachine) 
  (h1 : machine.red = 12)
  (h2 : machine.white = 10)
  (h3 : machine.blue = 9)
  (h4 : machine.green = 8) : 
  minGumballsForFourSame machine = 13 := by
  sorry

end min_gumballs_for_four_same_is_thirteen_l1386_138611


namespace circle_numbers_solution_l1386_138664

def CircleNumbers (a b c d e f : ℚ) : Prop :=
  a + b + c + d + e + f = 1 ∧
  a = |b - c| ∧
  b = |c - d| ∧
  c = |d - e| ∧
  d = |e - f| ∧
  e = |f - a| ∧
  f = |a - b|

theorem circle_numbers_solution :
  ∀ a b c d e f : ℚ, CircleNumbers a b c d e f →
  ((a = 1/4 ∧ b = 1/4 ∧ c = 0 ∧ d = 1/4 ∧ e = 1/4 ∧ f = 0) ∨
   (a = 1/4 ∧ b = 0 ∧ c = 1/4 ∧ d = 1/4 ∧ e = 0 ∧ f = 1/4) ∨
   (a = 0 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 0 ∧ e = 1/4 ∧ f = 1/4)) :=
by sorry

end circle_numbers_solution_l1386_138664


namespace smallest_student_group_l1386_138698

theorem smallest_student_group (n : ℕ) : n = 46 ↔ 
  (n > 0) ∧ 
  (n % 3 = 1) ∧ 
  (n % 6 = 4) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 1 → m % 6 = 4 → m % 8 = 5 → m ≥ n) :=
by sorry

end smallest_student_group_l1386_138698


namespace quadratic_equations_solutions_l1386_138606

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 3 * x^2 - 6 * x - 2 = 0 ↔ x = 1 + Real.sqrt 15 / 3 ∨ x = 1 - Real.sqrt 15 / 3) ∧
  (∀ x : ℝ, x^2 - 2 - 3 * x = 0 ↔ x = (3 + Real.sqrt 17) / 2 ∨ x = (3 - Real.sqrt 17) / 2) :=
by sorry

end quadratic_equations_solutions_l1386_138606


namespace expression_simplification_l1386_138653

theorem expression_simplification (a b : ℤ) : 
  (a = 1) → (b = -a) → 3*a^2*b + 2*(a*b - 3/2*a^2*b) - (2*a*b^2 - (3*a*b^2 - a*b)) = 0 := by
  sorry

end expression_simplification_l1386_138653


namespace sum_of_W_and_Y_l1386_138678

def problem (W X Y Z : ℕ) : Prop :=
  W ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  X ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  Y ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  Z ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  (W * X : ℚ) / (Y * Z) + (Y : ℚ) / Z = 3

theorem sum_of_W_and_Y (W X Y Z : ℕ) :
  problem W X Y Z → W + Y = 8 :=
by sorry

end sum_of_W_and_Y_l1386_138678


namespace gp_common_ratio_l1386_138609

/-- 
Theorem: In a geometric progression where the ratio of the sum of the first 6 terms 
to the sum of the first 3 terms is 217, the common ratio is 6.
-/
theorem gp_common_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217 → r = 6 := by
  sorry

end gp_common_ratio_l1386_138609


namespace root_range_implies_k_range_l1386_138672

theorem root_range_implies_k_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, 
    x₁^2 + (k-3)*x₁ + k^2 = 0 ∧
    x₂^2 + (k-3)*x₂ + k^2 = 0 ∧
    x₁ < 1 ∧ x₂ > 1 ∧ x₁ ≠ x₂) →
  k > -2 ∧ k < 1 :=
by sorry

end root_range_implies_k_range_l1386_138672


namespace cos_48_degrees_l1386_138648

theorem cos_48_degrees : Real.cos (48 * π / 180) = Real.cos (48 * π / 180) := by
  sorry

end cos_48_degrees_l1386_138648


namespace expression_evaluation_l1386_138619

theorem expression_evaluation : (100 - (5000 - 500)) * (5000 - (500 - 100)) = -20240000 := by
  sorry

end expression_evaluation_l1386_138619


namespace marcy_lip_gloss_tubs_l1386_138663

/-- The number of tubs of lip gloss Marcy needs to bring for a wedding -/
def tubs_of_lip_gloss (people : ℕ) (people_per_tube : ℕ) (tubes_per_tub : ℕ) : ℕ :=
  (people / people_per_tube) / tubes_per_tub

/-- Theorem: Marcy needs to bring 6 tubs of lip gloss for 36 people -/
theorem marcy_lip_gloss_tubs : tubs_of_lip_gloss 36 3 2 = 6 := by
  sorry

end marcy_lip_gloss_tubs_l1386_138663


namespace fraction_calculation_l1386_138602

theorem fraction_calculation (x y : ℚ) (hx : x = 4/6) (hy : y = 8/12) :
  (6*x + 8*y) / (48*x*y) = 7/16 := by sorry

end fraction_calculation_l1386_138602


namespace picture_book_shelves_l1386_138695

theorem picture_book_shelves 
  (total_books : ℕ) 
  (books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) 
  (h1 : total_books = 72)
  (h2 : books_per_shelf = 9)
  (h3 : mystery_shelves = 3) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 5 := by
sorry

end picture_book_shelves_l1386_138695


namespace cube_root_existence_l1386_138605

theorem cube_root_existence : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := by
  sorry

end cube_root_existence_l1386_138605


namespace intersection_line_equation_l1386_138651

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line_equation x y := by
  sorry

end intersection_line_equation_l1386_138651


namespace unique_dot_product_solution_l1386_138670

theorem unique_dot_product_solution (a : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ 
    (-Real.sin x * Real.sin (3 * x) + Real.sin (2 * x) * Real.sin (4 * x) = a)) → 
  a = 1 := by
sorry

end unique_dot_product_solution_l1386_138670
