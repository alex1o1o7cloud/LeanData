import Mathlib

namespace NUMINAMATH_CALUDE_rowing_distance_with_tide_l3635_363524

/-- Represents the problem of a man rowing with and against the tide. -/
structure RowingProblem where
  /-- The speed of the man rowing in still water (km/h) -/
  manSpeed : ℝ
  /-- The speed of the tide (km/h) -/
  tideSpeed : ℝ
  /-- The distance traveled against the tide (km) -/
  distanceAgainstTide : ℝ
  /-- The time taken to travel against the tide (h) -/
  timeAgainstTide : ℝ
  /-- The time that would have been saved if the tide hadn't changed (h) -/
  timeSaved : ℝ

/-- Theorem stating that given the conditions of the rowing problem, 
    the distance the man can row with the help of the tide in 60 minutes is 5 km. -/
theorem rowing_distance_with_tide (p : RowingProblem) 
  (h1 : p.manSpeed - p.tideSpeed = p.distanceAgainstTide / p.timeAgainstTide)
  (h2 : p.manSpeed + p.tideSpeed = p.distanceAgainstTide / (p.timeAgainstTide - p.timeSaved))
  (h3 : p.distanceAgainstTide = 40)
  (h4 : p.timeAgainstTide = 10)
  (h5 : p.timeSaved = 2) :
  (p.manSpeed + p.tideSpeed) * 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rowing_distance_with_tide_l3635_363524


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_half_sum_of_other_squares_l3635_363595

theorem sum_of_squares_equals_half_sum_of_other_squares (a b : ℝ) :
  a^2 + b^2 = ((a + b)^2 + (a - b)^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_half_sum_of_other_squares_l3635_363595


namespace NUMINAMATH_CALUDE_alice_bob_number_sum_l3635_363502

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B →
    (¬(is_prime A) ∧ ¬(¬is_prime A)) →
    (¬(is_prime B) ∧ is_perfect_square B) →
    is_perfect_square (50 * B + A) →
    A + B = 43 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_number_sum_l3635_363502


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l3635_363581

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l3635_363581


namespace NUMINAMATH_CALUDE_table_tennis_pairing_methods_l3635_363506

theorem table_tennis_pairing_methods (total_players : Nat) (male_players : Nat) (female_players : Nat) :
  total_players = male_players + female_players →
  male_players = 5 →
  female_players = 4 →
  (Nat.choose male_players 2) * (Nat.choose female_players 2) * 2 = 120 :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_pairing_methods_l3635_363506


namespace NUMINAMATH_CALUDE_dodecagon_square_area_ratio_l3635_363567

theorem dodecagon_square_area_ratio :
  ∀ (square_side : ℝ) (dodecagon_area : ℝ),
    square_side = 2 →
    dodecagon_area = 3 →
    ∃ (shaded_area : ℝ),
      shaded_area = (square_side^2 - dodecagon_area) / 4 ∧
      shaded_area / dodecagon_area = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_square_area_ratio_l3635_363567


namespace NUMINAMATH_CALUDE_birdseed_solution_l3635_363599

/-- The number of boxes of birdseed Leah already had in the pantry -/
def birdseed_problem (new_boxes : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) 
  (box_content : ℕ) (weeks : ℕ) : ℕ :=
  let total_consumption := parrot_consumption + cockatiel_consumption
  let total_needed := total_consumption * weeks
  let total_boxes := (total_needed + box_content - 1) / box_content
  total_boxes - new_boxes

/-- Theorem stating the solution to the birdseed problem -/
theorem birdseed_solution : 
  birdseed_problem 3 100 50 225 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_birdseed_solution_l3635_363599


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l3635_363580

theorem ancient_chinese_math_problem (x y : ℚ) : 
  8 * x = y + 3 ∧ 7 * x = y - 4 → (y + 3) / 8 = (y - 4) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l3635_363580


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3635_363546

-- Define the number of physics and history books
def num_physics_books : ℕ := 4
def num_history_books : ℕ := 6

-- Define the function to calculate the number of arrangements
def num_arrangements (p h : ℕ) : ℕ :=
  2 * (Nat.factorial p) * (Nat.factorial h)

-- Theorem statement
theorem book_arrangement_count :
  num_arrangements num_physics_books num_history_books = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3635_363546


namespace NUMINAMATH_CALUDE_quadratic_sum_l3635_363574

/-- Given a quadratic polynomial 6x^2 + 36x + 216, when expressed in the form a(x + b)^2 + c,
    where a, b, and c are constants, prove that a + b + c = 171. -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3635_363574


namespace NUMINAMATH_CALUDE_time_difference_walk_vs_bicycle_l3635_363582

/-- Represents the number of blocks from Henrikh's home to his office -/
def distance : ℕ := 12

/-- Represents the time in minutes to walk one block -/
def walkingTimePerBlock : ℚ := 1

/-- Represents the time in minutes to ride a bicycle for one block -/
def bicycleTimePerBlock : ℚ := 20 / 60

/-- Calculates the total time to travel the distance by walking -/
def walkingTime : ℚ := distance * walkingTimePerBlock

/-- Calculates the total time to travel the distance by bicycle -/
def bicycleTime : ℚ := distance * bicycleTimePerBlock

theorem time_difference_walk_vs_bicycle :
  walkingTime - bicycleTime = 8 := by sorry

end NUMINAMATH_CALUDE_time_difference_walk_vs_bicycle_l3635_363582


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l3635_363520

theorem divisibility_of_polynomial (n : ℤ) : 
  (120 : ℤ) ∣ (n^5 - 5*n^3 + 4*n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l3635_363520


namespace NUMINAMATH_CALUDE_perpendicular_planes_l3635_363584

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relationship between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relationship between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relationship between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define non-overlapping relationship for lines
variable (non_overlapping_lines : Line → Line → Prop)

-- Define non-overlapping relationship for planes
variable (non_overlapping_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane)
  (h1 : non_overlapping_lines m n)
  (h2 : non_overlapping_planes α β)
  (h3 : perp_line_plane m α)
  (h4 : perp_line_plane n β)
  (h5 : perp_line_line m n) :
  perp_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l3635_363584


namespace NUMINAMATH_CALUDE_geometric_sequence_divisibility_l3635_363540

/-- Given a geometric sequence with first term a₁ and second term a₂, 
    find the smallest n for which the n-th term is divisible by 10⁶ -/
theorem geometric_sequence_divisibility
  (a₁ : ℚ)
  (a₂ : ℕ)
  (h₁ : a₁ = 5 / 8)
  (h₂ : a₂ = 25)
  : ∃ n : ℕ, n > 0 ∧ 
    (∀ k < n, ¬(10^6 ∣ (a₁ * (a₂ / a₁)^(k - 1)))) ∧
    (10^6 ∣ (a₁ * (a₂ / a₁)^(n - 1))) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_divisibility_l3635_363540


namespace NUMINAMATH_CALUDE_baseball_cards_cost_theorem_l3635_363541

/-- The cost of a baseball card deck given the total spent and the cost of Digimon card packs -/
def baseball_card_cost (total_spent : ℝ) (digimon_pack_cost : ℝ) (num_digimon_packs : ℕ) : ℝ :=
  total_spent - (digimon_pack_cost * num_digimon_packs)

/-- Theorem: The cost of the baseball cards is $6.06 -/
theorem baseball_cards_cost_theorem (total_spent : ℝ) (digimon_pack_cost : ℝ) (num_digimon_packs : ℕ) :
  total_spent = 23.86 ∧ digimon_pack_cost = 4.45 ∧ num_digimon_packs = 4 →
  baseball_card_cost total_spent digimon_pack_cost num_digimon_packs = 6.06 := by
  sorry

#eval baseball_card_cost 23.86 4.45 4

end NUMINAMATH_CALUDE_baseball_cards_cost_theorem_l3635_363541


namespace NUMINAMATH_CALUDE_chad_savings_l3635_363555

/-- Represents Chad's financial situation for a year --/
structure ChadFinances where
  savingRate : ℝ
  mowingIncome : ℝ
  birthdayMoney : ℝ
  videoGamesSales : ℝ
  oddJobsIncome : ℝ

/-- Calculates Chad's total savings for the year --/
def totalSavings (cf : ChadFinances) : ℝ :=
  cf.savingRate * (cf.mowingIncome + cf.birthdayMoney + cf.videoGamesSales + cf.oddJobsIncome)

/-- Theorem stating that Chad's savings for the year will be $460 --/
theorem chad_savings :
  ∀ (cf : ChadFinances),
    cf.savingRate = 0.4 ∧
    cf.mowingIncome = 600 ∧
    cf.birthdayMoney = 250 ∧
    cf.videoGamesSales = 150 ∧
    cf.oddJobsIncome = 150 →
    totalSavings cf = 460 :=
by sorry

end NUMINAMATH_CALUDE_chad_savings_l3635_363555


namespace NUMINAMATH_CALUDE_revenue_decrease_percent_l3635_363519

theorem revenue_decrease_percent (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let R := T * C
  let T_new := T * (1 - 0.20)
  let C_new := C * (1 + 0.15)
  let R_new := T_new * C_new
  (R - R_new) / R * 100 = 8 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_percent_l3635_363519


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l3635_363518

/-- Calculates the number of hours Amy needs to work per week during the school year --/
def school_year_hours_per_week (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℚ :=
  let summer_total_hours := summer_hours_per_week * summer_weeks
  let hourly_rate := summer_earnings / summer_total_hours
  let school_year_total_hours := school_year_earnings / hourly_rate
  school_year_total_hours / school_year_weeks

/-- Theorem stating that Amy needs to work 9 hours per week during the school year --/
theorem amy_school_year_hours : 
  school_year_hours_per_week 36 10 3000 40 3000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_school_year_hours_l3635_363518


namespace NUMINAMATH_CALUDE_tan_BAC_equals_three_fourths_l3635_363535

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D and E on sides AB and AC
structure TriangleWithDE extends Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (D_on_AB : D.1 = A.1 + t * (B.1 - A.1) ∧ D.2 = A.2 + t * (B.2 - A.2)) 
  (E_on_AC : E.1 = A.1 + s * (C.1 - A.1) ∧ E.2 = A.2 + s * (C.2 - A.2))
  (t s : ℝ)
  (t_range : 0 < t ∧ t < 1)
  (s_range : 0 < s ∧ s < 1)

-- Define the area of triangle ADE
def area_ADE (t : TriangleWithDE) : ℝ := sorry

-- Define the incircle of quadrilateral BDEC
structure Incircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point K where the incircle touches AB
def point_K (t : TriangleWithDE) (i : Incircle) : ℝ × ℝ := sorry

-- Define the function to calculate tan(BAC)
def tan_BAC (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem tan_BAC_equals_three_fourths 
  (t : TriangleWithDE) 
  (i : Incircle) 
  (h1 : area_ADE t = 0.5)
  (h2 : point_K t i = (t.A.1 + 3, t.A.2))
  (h3 : (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2 = 15^2)
  (h4 : ∃ (center : ℝ × ℝ) (radius : ℝ), 
        (t.B.1 - center.1)^2 + (t.B.2 - center.2)^2 = radius^2 ∧
        (t.D.1 - center.1)^2 + (t.D.2 - center.2)^2 = radius^2 ∧
        (t.E.1 - center.1)^2 + (t.E.2 - center.2)^2 = radius^2 ∧
        (t.C.1 - center.1)^2 + (t.C.2 - center.2)^2 = radius^2) :
  tan_BAC t.toTriangle = 3/4 := by sorry

end NUMINAMATH_CALUDE_tan_BAC_equals_three_fourths_l3635_363535


namespace NUMINAMATH_CALUDE_triangle_area_is_36_l3635_363528

/-- The area of the triangle bounded by y = x, y = -x, and y = 6 -/
def triangle_area : ℝ := 36

/-- The line y = x -/
def line1 (x : ℝ) : ℝ := x

/-- The line y = -x -/
def line2 (x : ℝ) : ℝ := -x

/-- The line y = 6 -/
def line3 : ℝ := 6

theorem triangle_area_is_36 :
  triangle_area = 36 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_36_l3635_363528


namespace NUMINAMATH_CALUDE_chess_club_girls_count_l3635_363534

theorem chess_club_girls_count (total_members : ℕ) (present_members : ℕ) 
  (h1 : total_members = 32)
  (h2 : present_members = 20)
  (h3 : ∃ (boys girls : ℕ), boys + girls = total_members ∧ boys + girls / 2 = present_members) :
  ∃ (girls : ℕ), girls = 24 ∧ ∃ (boys : ℕ), boys + girls = total_members := by
sorry

end NUMINAMATH_CALUDE_chess_club_girls_count_l3635_363534


namespace NUMINAMATH_CALUDE_lisas_age_2005_l3635_363556

theorem lisas_age_2005 (lisa_age_2000 grandfather_age_2000 : ℕ) 
  (h1 : lisa_age_2000 * 2 = grandfather_age_2000)
  (h2 : (2000 - lisa_age_2000) + (2000 - grandfather_age_2000) = 3904) :
  lisa_age_2000 + 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_lisas_age_2005_l3635_363556


namespace NUMINAMATH_CALUDE_complex_sixth_power_equation_simplified_polynomial_system_l3635_363568

/-- The complex number z satisfying z^6 = -8 - 8i can be characterized by a system of polynomial equations. -/
theorem complex_sixth_power_equation (z : ℂ) : 
  z^6 = -8 - 8*I ↔ 
  ∃ (x y : ℝ), z = x + y*I ∧ 
    (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8) ∧
    (6*x^5*y - 20*x^3*y^3 + 6*x*y^5 = -8) :=
by sorry

/-- The system of polynomial equations characterizing the solutions can be further simplified. -/
theorem simplified_polynomial_system (x y : ℝ) :
  (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8 ∧ 
   6*x^5*y - 20*x^3*y^3 + 6*x*y^5 = -8) ↔
  (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8 ∧
   x^4 - 10*x^2*y^2 + y^4 = -4/3) :=
by sorry

end NUMINAMATH_CALUDE_complex_sixth_power_equation_simplified_polynomial_system_l3635_363568


namespace NUMINAMATH_CALUDE_evaluate_expression_l3635_363592

theorem evaluate_expression : -(18 / 3 * 8 - 72 + 4 * 8) = 8 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3635_363592


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3635_363579

theorem roots_sum_of_squares (a b : ℝ) : 
  a^2 - a - 2023 = 0 → b^2 - b - 2023 = 0 → a^2 + b^2 = 4047 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3635_363579


namespace NUMINAMATH_CALUDE_remainder_2984_times_3998_mod_1000_l3635_363514

theorem remainder_2984_times_3998_mod_1000 : (2984 * 3998) % 1000 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2984_times_3998_mod_1000_l3635_363514


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3635_363585

theorem indefinite_integral_proof (x : ℝ) : 
  (deriv (λ x => -1/4 * (7*x - 10) * Real.cos (4*x) - 7/16 * Real.sin (4*x))) x = 
  (7*x - 10) * Real.sin (4*x) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3635_363585


namespace NUMINAMATH_CALUDE_exists_vertical_line_through_point_l3635_363525

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : Option ℝ
  yIntercept : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  match l.slope with
  | some k => p.y = k * p.x + l.yIntercept
  | none => p.x = l.yIntercept

-- Theorem statement
theorem exists_vertical_line_through_point (b : ℝ) :
  ∃ (l : Line2D), pointOnLine ⟨0, b⟩ l ∧ l.slope = none :=
sorry

end NUMINAMATH_CALUDE_exists_vertical_line_through_point_l3635_363525


namespace NUMINAMATH_CALUDE_cubic_expression_lower_bound_l3635_363551

theorem cubic_expression_lower_bound (x : ℝ) (h : x^2 - 5*x + 6 > 0) :
  x^3 - 5*x^2 + 6*x + 1 ≥ 1 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_lower_bound_l3635_363551


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l3635_363522

/-- Revenue function for book sales -/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The optimal price maximizes revenue -/
theorem optimal_price_maximizes_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue p ≥ revenue q ∧
  p = 19 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l3635_363522


namespace NUMINAMATH_CALUDE_choose_cooks_count_l3635_363587

def total_people : ℕ := 10
def cooks_needed : ℕ := 3

theorem choose_cooks_count : Nat.choose total_people cooks_needed = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_cooks_count_l3635_363587


namespace NUMINAMATH_CALUDE_sum_of_quotients_divisible_by_nine_l3635_363554

theorem sum_of_quotients_divisible_by_nine (n : ℕ) (hn : n > 8) :
  let a : ℕ → ℕ := λ i => (10^(2*i) - 1) / 9
  let q : ℕ → ℕ := λ i => a i / 11
  let s : ℕ → ℕ := λ i => (Finset.range 9).sum (λ j => q (i + j))
  ∀ i : ℕ, i ≤ n - 8 → (s i) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quotients_divisible_by_nine_l3635_363554


namespace NUMINAMATH_CALUDE_impossibility_of_circular_arrangement_l3635_363591

theorem impossibility_of_circular_arrangement : ¬ ∃ (arrangement : Fin 1995 → ℕ), 
  (∀ i j : Fin 1995, i ≠ j → arrangement i ≠ arrangement j) ∧ 
  (∀ i : Fin 1995, Nat.Prime ((max (arrangement i) (arrangement (i + 1))) / 
                               (min (arrangement i) (arrangement (i + 1))))) :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_circular_arrangement_l3635_363591


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_2alpha_l3635_363583

theorem parallel_vectors_tan_2alpha (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi)
  (h2 : (Real.cos α - 5) * Real.cos α + Real.sin α * (Real.sin α - 5) = 0) :
  Real.tan (2 * α) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_2alpha_l3635_363583


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l3635_363590

theorem president_and_committee_selection (n : ℕ) (h : n = 10) : 
  n * (Nat.choose (n - 1) 3) = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_selection_l3635_363590


namespace NUMINAMATH_CALUDE_sales_growth_rate_equation_l3635_363529

/-- The average monthly growth rate of a store's sales revenue -/
def average_monthly_growth_rate (march_revenue : ℝ) (may_revenue : ℝ) : ℝ → Prop :=
  λ x => 3 * (1 + x)^2 = 3.63

theorem sales_growth_rate_equation (march_revenue may_revenue : ℝ) 
  (h1 : march_revenue = 30000)
  (h2 : may_revenue = 36300) :
  ∃ x, average_monthly_growth_rate march_revenue may_revenue x :=
by
  sorry

end NUMINAMATH_CALUDE_sales_growth_rate_equation_l3635_363529


namespace NUMINAMATH_CALUDE_cone_height_l3635_363572

theorem cone_height (r : ℝ) (lateral_area : ℝ) (h : ℝ) : 
  r = 1 → lateral_area = 2 * Real.pi → h = Real.sqrt 3 → 
  lateral_area = Real.pi * r * Real.sqrt (h^2 + r^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l3635_363572


namespace NUMINAMATH_CALUDE_brad_carl_weight_difference_l3635_363570

/-- Given the weights of Billy, Brad, and Carl, prove that Brad weighs 5 pounds more than Carl. -/
theorem brad_carl_weight_difference
  (billy_weight : ℕ)
  (brad_weight : ℕ)
  (carl_weight : ℕ)
  (h1 : billy_weight = brad_weight + 9)
  (h2 : brad_weight > carl_weight)
  (h3 : carl_weight = 145)
  (h4 : billy_weight = 159) :
  brad_weight - carl_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_brad_carl_weight_difference_l3635_363570


namespace NUMINAMATH_CALUDE_exactly_two_true_l3635_363516

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define equilateral triangle
def equilateral (t : Triangle) : Prop := 
  ∀ i : Fin 3, t.angles i = 60

-- Proposition 1
def prop1 : Prop := 
  ∀ t1 t2 : Triangle, t1.area = t2.area → congruent t1 t2

-- Proposition 2
def prop2 : Prop := 
  ∃ a b : ℝ, a * b = 0 ∧ a ≠ 0

-- Proposition 3
def prop3 : Prop := 
  ∀ t : Triangle, ¬equilateral t → ∃ i : Fin 3, t.angles i ≠ 60

-- Main theorem
theorem exactly_two_true : 
  (¬prop1 ∧ prop2 ∧ prop3) ∨
  (prop1 ∧ prop2 ∧ ¬prop3) ∨
  (prop1 ∧ ¬prop2 ∧ prop3) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_true_l3635_363516


namespace NUMINAMATH_CALUDE_team_omega_score_l3635_363578

/-- Given a basketball match between Team Alpha and Team Omega where:
  - The total points scored by both teams is 60
  - Team Alpha won by a margin of 12 points
  This theorem proves that Team Omega scored 24 points. -/
theorem team_omega_score (total_points : ℕ) (margin : ℕ) 
  (h1 : total_points = 60) 
  (h2 : margin = 12) : 
  (total_points - margin) / 2 = 24 := by
  sorry

#check team_omega_score

end NUMINAMATH_CALUDE_team_omega_score_l3635_363578


namespace NUMINAMATH_CALUDE_equation_solution_l3635_363537

theorem equation_solution (a : ℝ) (h : a = 0.5) : 
  ∃ x : ℝ, x / (a - 3) = 3 / (a + 2) ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3635_363537


namespace NUMINAMATH_CALUDE_fraction_comparison_l3635_363508

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3635_363508


namespace NUMINAMATH_CALUDE_ping_pong_ball_price_l3635_363586

theorem ping_pong_ball_price 
  (quantity : ℕ) 
  (discount_rate : ℚ) 
  (total_paid : ℚ) 
  (h1 : quantity = 10000)
  (h2 : discount_rate = 30 / 100)
  (h3 : total_paid = 700) :
  let original_price := total_paid / ((1 - discount_rate) * quantity)
  original_price = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_ping_pong_ball_price_l3635_363586


namespace NUMINAMATH_CALUDE_crayon_difference_l3635_363503

def karen_crayons : ℕ := 639
def cindy_crayons : ℕ := 504
def peter_crayons : ℕ := 752
def rachel_crayons : ℕ := 315

theorem crayon_difference :
  (max karen_crayons (max cindy_crayons (max peter_crayons rachel_crayons))) -
  (min karen_crayons (min cindy_crayons (min peter_crayons rachel_crayons))) = 437 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l3635_363503


namespace NUMINAMATH_CALUDE_investment_amount_l3635_363515

/-- Proves that given a monthly interest payment of $240 and a simple annual interest rate of 9%,
    the principal amount of the investment is $32,000. -/
theorem investment_amount (monthly_interest : ℝ) (annual_rate : ℝ) (principal : ℝ) :
  monthly_interest = 240 →
  annual_rate = 0.09 →
  principal = monthly_interest / (annual_rate / 12) →
  principal = 32000 := by
  sorry

end NUMINAMATH_CALUDE_investment_amount_l3635_363515


namespace NUMINAMATH_CALUDE_negative_comparison_l3635_363564

theorem negative_comparison : -2023 > -2024 := by
  sorry

end NUMINAMATH_CALUDE_negative_comparison_l3635_363564


namespace NUMINAMATH_CALUDE_solve_equation_l3635_363521

theorem solve_equation (A : ℝ) : 3 + A = 4 → A = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3635_363521


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3635_363558

/-- A geometric sequence of positive real numbers -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 4) →
  (a 5 + a 6 = 16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3635_363558


namespace NUMINAMATH_CALUDE_max_stones_upper_bound_max_stones_achievable_max_stones_theorem_l3635_363593

/-- Represents the state of the piles -/
def PileState := List Nat

/-- The initial state of the piles -/
def initial_state : PileState := List.replicate 2009 2

/-- The operation of transferring stones -/
def transfer (state : PileState) : PileState :=
  sorry

/-- Predicate to check if a state is valid -/
def is_valid_state (state : PileState) : Prop :=
  state.sum = 2009 * 2 ∧ state.all (· ≥ 1)

/-- The maximum number of stones in any pile -/
def max_stones (state : PileState) : Nat :=
  state.foldl Nat.max 0

theorem max_stones_upper_bound :
  ∀ (state : PileState), is_valid_state state → max_stones state ≤ 2010 :=
  sorry

theorem max_stones_achievable :
  ∃ (state : PileState), is_valid_state state ∧ max_stones state = 2010 :=
  sorry

theorem max_stones_theorem :
  (∀ (state : PileState), is_valid_state state → max_stones state ≤ 2010) ∧
  (∃ (state : PileState), is_valid_state state ∧ max_stones state = 2010) :=
  sorry

end NUMINAMATH_CALUDE_max_stones_upper_bound_max_stones_achievable_max_stones_theorem_l3635_363593


namespace NUMINAMATH_CALUDE_lightest_pumpkin_weight_l3635_363553

/-- Given three pumpkins with weights A, B, and C, prove that A = 5 -/
theorem lightest_pumpkin_weight (A B C : ℕ) 
  (h1 : A ≤ B) (h2 : B ≤ C)
  (h3 : A + B = 12) (h4 : A + C = 13) (h5 : B + C = 15) : 
  A = 5 := by
  sorry

end NUMINAMATH_CALUDE_lightest_pumpkin_weight_l3635_363553


namespace NUMINAMATH_CALUDE_customers_who_tipped_l3635_363571

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ)
  (h1 : initial_customers = 29)
  (h2 : additional_customers = 20)
  (h3 : non_tipping_customers = 34) :
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l3635_363571


namespace NUMINAMATH_CALUDE_concatenated_number_divisibility_l3635_363523

theorem concatenated_number_divisibility
  (n : ℕ) (a : ℕ) (h_n : n > 1) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) :
  let b := a * 10^n + a
  (∃ d : ℕ, b = d * a^2) → b / a^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_concatenated_number_divisibility_l3635_363523


namespace NUMINAMATH_CALUDE_sugar_calculation_l3635_363517

-- Define the original amount of sugar in the recipe
def original_sugar : ℚ := 5 + 3/4

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/3

-- Define the result we want to prove
def result : ℚ := 1 + 11/12

-- Theorem statement
theorem sugar_calculation : 
  recipe_fraction * original_sugar = result := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l3635_363517


namespace NUMINAMATH_CALUDE_x_value_for_purely_imaginary_square_l3635_363544

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem x_value_for_purely_imaginary_square (x : ℝ) :
  x > 0 → isPurelyImaginary ((x - complex 0 1) ^ 2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_for_purely_imaginary_square_l3635_363544


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3635_363505

theorem distinct_prime_factors_count (n : ℕ) : n = 95 * 97 * 99 * 101 → Finset.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3635_363505


namespace NUMINAMATH_CALUDE_min_printers_equal_expenditure_l3635_363573

def printer_costs : List Nat := [400, 350, 500, 200]

theorem min_printers_equal_expenditure :
  let total_cost := Nat.lcm (Nat.lcm (Nat.lcm 400 350) 500) 200
  let num_printers := List.sum (List.map (λ cost => total_cost / cost) printer_costs)
  num_printers = 173 ∧
  ∀ (n : Nat), n < num_printers →
    ∃ (cost : Nat), cost ∈ printer_costs ∧ (n * cost) % total_cost ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_printers_equal_expenditure_l3635_363573


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3635_363550

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The original rectangle -/
def original : Rectangle := { width := 5, height := 7 }

/-- The rectangle after shortening one side by 2 inches -/
def shortened : Rectangle := { width := 3, height := 7 }

/-- The rectangle after shortening the other side by 1 inch -/
def other_shortened : Rectangle := { width := 5, height := 6 }

theorem rectangle_area_theorem :
  original.area = 35 ∧
  shortened.area = 21 →
  other_shortened.area = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3635_363550


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_function_l3635_363598

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the property of f being even when shifted by 2
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x + 2) = f (x + 2)

-- Define the symmetry axis of a function
def symmetry_axis (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Theorem statement
theorem symmetry_of_shifted_function (f : ℝ → ℝ) 
  (h : is_even_shifted f) : 
  symmetry_axis (fun x ↦ f (x - 1) + 2) 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_function_l3635_363598


namespace NUMINAMATH_CALUDE_student_failed_marks_l3635_363511

theorem student_failed_marks (total_marks : ℕ) (passing_percentage : ℚ) (student_score : ℕ) : 
  total_marks = 600 → 
  passing_percentage = 33 / 100 → 
  student_score = 125 → 
  (total_marks * passing_percentage).floor - student_score = 73 := by
  sorry

end NUMINAMATH_CALUDE_student_failed_marks_l3635_363511


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3635_363588

/-- A polynomial satisfying the given functional equation -/
def FunctionalEquationPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

/-- The form of the polynomial satisfying the functional equation -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ n : ℕ+, ∀ x : ℝ, P x = (x + 1)^(n : ℕ)

/-- Theorem stating that any non-zero polynomial satisfying the functional equation
    must be of the form (x + 1)^n for some positive integer n -/
theorem functional_equation_solution :
  ∀ P : ℝ → ℝ, (∃ x : ℝ, P x ≠ 0) → FunctionalEquationPolynomial P → PolynomialForm P :=
by sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3635_363588


namespace NUMINAMATH_CALUDE_sqrt_30_between_5_and_6_l3635_363500

theorem sqrt_30_between_5_and_6 : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_30_between_5_and_6_l3635_363500


namespace NUMINAMATH_CALUDE_city_council_vote_change_l3635_363576

theorem city_council_vote_change :
  ∀ (x y x' y' : ℕ),
    x + y = 500 →
    y > x →
    x' + y' = 500 →
    x' - y' = (3 * (y - x)) / 2 →
    x' = (13 * y) / 12 →
    x' - x = 125 :=
by sorry

end NUMINAMATH_CALUDE_city_council_vote_change_l3635_363576


namespace NUMINAMATH_CALUDE_base_conversion_four_digits_l3635_363560

theorem base_conversion_four_digits (b : ℕ) : b > 1 → (
  (256 < b^4) ∧ (b^3 ≤ 256) ↔ b = 5
) := by sorry

end NUMINAMATH_CALUDE_base_conversion_four_digits_l3635_363560


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3635_363575

theorem sphere_surface_area (V : Real) (r : Real) : 
  V = (4 / 3) * Real.pi * r^3 → 
  V = 36 * Real.pi → 
  4 * Real.pi * r^2 = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3635_363575


namespace NUMINAMATH_CALUDE_factorization_problems_l3635_363577

theorem factorization_problems (a b x y : ℝ) : 
  (2 * x * (a - b) - (b - a) = (a - b) * (2 * x + 1)) ∧ 
  ((x^2 + y^2)^2 - 4 * x^2 * y^2 = (x - y)^2 * (x + y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l3635_363577


namespace NUMINAMATH_CALUDE_grocery_store_soda_count_l3635_363566

/-- Given a grocery store inventory, prove the number of regular soda bottles -/
theorem grocery_store_soda_count 
  (diet_soda : ℕ) 
  (regular_soda_diff : ℕ) 
  (h1 : diet_soda = 53)
  (h2 : regular_soda_diff = 26) :
  diet_soda + regular_soda_diff = 79 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_count_l3635_363566


namespace NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l3635_363549

theorem cos_alpha_plus_seven_pi_twelfths (α : ℝ) 
  (h : Real.sin (α + π / 12) = 1 / 3) : 
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l3635_363549


namespace NUMINAMATH_CALUDE_train_length_is_160_meters_l3635_363548

def train_speed : ℝ := 45 -- km/hr
def crossing_time : ℝ := 30 -- seconds
def bridge_length : ℝ := 215 -- meters

theorem train_length_is_160_meters :
  let speed_mps := train_speed * 1000 / 3600
  let total_distance := speed_mps * crossing_time
  total_distance - bridge_length = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_is_160_meters_l3635_363548


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3635_363513

theorem sum_of_x_and_y (x y : ℚ) 
  (hx : |x| = 5)
  (hy : |y| = 2)
  (hxy : |x - y| = x - y) :
  x + y = 7 ∨ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3635_363513


namespace NUMINAMATH_CALUDE_circle_plus_five_two_l3635_363547

/-- The custom binary operation ⊕ -/
def circle_plus (x y : ℝ) : ℝ := (x + y + 1) * (x - y)

/-- Theorem stating that 5 ⊕ 2 = 24 -/
theorem circle_plus_five_two : circle_plus 5 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_five_two_l3635_363547


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3635_363510

theorem fraction_equation_solution : ∃ x : ℚ, (1 / 2 - 1 / 3 : ℚ) = 1 / x ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3635_363510


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3635_363545

theorem fraction_to_decimal : (22 : ℚ) / 160 = (1375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3635_363545


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3635_363563

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we want to express in scientific notation -/
def original_number : ℕ := 135000

/-- The proposed scientific notation representation -/
def scientific_form : ScientificNotation := {
  coefficient := 1.35
  exponent := 5
  coeff_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3635_363563


namespace NUMINAMATH_CALUDE_system_solutions_l3635_363539

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  y = 2 * x^2 - 1 ∧ z = 2 * y^2 - 1 ∧ x = 2 * z^2 - 1

/-- The set of solutions to the system -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(1, 1, 1), (-1/2, -1/2, -1/2)} ∪
  {(Real.cos (2 * Real.pi / 9), Real.cos (4 * Real.pi / 9), -Real.cos (Real.pi / 9)),
   (Real.cos (4 * Real.pi / 9), -Real.cos (Real.pi / 9), Real.cos (2 * Real.pi / 9)),
   (-Real.cos (Real.pi / 9), Real.cos (2 * Real.pi / 9), Real.cos (4 * Real.pi / 9))} ∪
  {(Real.cos (2 * Real.pi / 7), -Real.cos (3 * Real.pi / 7), -Real.cos (Real.pi / 7)),
   (-Real.cos (3 * Real.pi / 7), -Real.cos (Real.pi / 7), Real.cos (2 * Real.pi / 7)),
   (-Real.cos (Real.pi / 7), Real.cos (2 * Real.pi / 7), -Real.cos (3 * Real.pi / 7))}

/-- Theorem stating that the solutions set contains all and only the solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℝ, (x, y, z) ∈ solutions ↔ system x y z :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l3635_363539


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_8_12_16_l3635_363509

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

/-- Theorem: The set (8, 12, 16) is not a Pythagorean triple -/
theorem not_pythagorean_triple_8_12_16 :
  ¬ is_pythagorean_triple 8 12 16 := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_8_12_16_l3635_363509


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3635_363597

theorem geometric_sequence_fourth_term (a b c x : ℝ) : 
  a ≠ 0 → b / a = c / b → c / b * c = x → a = 0.001 → b = 0.02 → c = 0.4 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3635_363597


namespace NUMINAMATH_CALUDE_problem_solution_l3635_363569

theorem problem_solution (t : ℝ) :
  let x := 3 - t
  let y := 2*t + 11
  x = 1 → y = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3635_363569


namespace NUMINAMATH_CALUDE_money_exchange_solution_money_exchange_unique_l3635_363538

/-- Represents the money exchange process between three friends -/
def money_exchange (a b c : ℕ) : Prop :=
  let step1_1 := a - b - c
  let step1_2 := 2 * b
  let step1_3 := 2 * c
  let step2_1 := 2 * (a - b - c)
  let step2_2 := 3 * b - a - 3 * c
  let step2_3 := 4 * c
  let step3_1 := 4 * (a - b - c)
  let step3_2 := 6 * b - 2 * a - 6 * c
  let step3_3 := 4 * c - 2 * (a - b - c) - (3 * b - a - 3 * c)
  step3_1 = 8 ∧ step3_2 = 8 ∧ step3_3 = 8

/-- Theorem stating that the initial amounts of 13, 7, and 4 écus result in each friend having 8 écus after the exchanges -/
theorem money_exchange_solution :
  money_exchange 13 7 4 :=
sorry

/-- Theorem stating that 13, 7, and 4 are the only initial amounts that result in each friend having 8 écus after the exchanges -/
theorem money_exchange_unique :
  ∀ a b c : ℕ, money_exchange a b c → (a = 13 ∧ b = 7 ∧ c = 4) :=
sorry

end NUMINAMATH_CALUDE_money_exchange_solution_money_exchange_unique_l3635_363538


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repetition_l3635_363504

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of digits in the numbers we're forming -/
def num_places : ℕ := 3

/-- The number of non-zero digits available for the first place -/
def non_zero_digits : ℕ := num_digits - 1

/-- The total number of three-digit numbers (including those without repetition) -/
def total_numbers : ℕ := non_zero_digits * num_digits * num_digits

/-- The number of three-digit numbers without repetition -/
def numbers_without_repetition : ℕ := non_zero_digits * (num_digits - 1) * (num_digits - 2)

theorem three_digit_numbers_with_repetition :
  total_numbers - numbers_without_repetition = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repetition_l3635_363504


namespace NUMINAMATH_CALUDE_m_range_l3635_363543

-- Define propositions P and Q
def P (m : ℝ) : Prop := |m + 1| ≤ 2
def Q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - m*x + 1 = 0

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, ¬(¬(P m))) →
  (∀ m : ℝ, ¬(P m ∧ Q m)) →
  ∀ m : ℝ, (m > -2 ∧ m ≤ 1) ↔ (P m ∧ ¬(Q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3635_363543


namespace NUMINAMATH_CALUDE_sum_of_distinct_numbers_l3635_363557

theorem sum_of_distinct_numbers (x y u v : ℝ) : 
  x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ y ≠ u ∧ y ≠ v ∧ u ≠ v →
  (x + u) / (x + v) = (y + v) / (y + u) →
  x + y + u + v = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_numbers_l3635_363557


namespace NUMINAMATH_CALUDE_box_max_volume_l3635_363596

/-- Volume function for the box -/
def V (x : ℝ) : ℝ := (16 - 2*x) * (10 - 2*x) * x

/-- The theorem stating the maximum volume and corresponding height -/
theorem box_max_volume :
  ∃ (max_vol : ℝ) (max_height : ℝ),
    (∀ x, 0 < x → x < 5 → V x ≤ max_vol) ∧
    (0 < max_height ∧ max_height < 5) ∧
    (V max_height = max_vol) ∧
    (max_height = 2) ∧
    (max_vol = 144) := by
  sorry

end NUMINAMATH_CALUDE_box_max_volume_l3635_363596


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l3635_363565

theorem complex_quadratic_roots : 
  ∃ (z₁ z₂ : ℂ), z₁ = Complex.I ∧ z₂ = -3 - 2*Complex.I ∧
  (∀ z : ℂ, z^2 + 2*z = -3 + 4*Complex.I ↔ z = z₁ ∨ z = z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l3635_363565


namespace NUMINAMATH_CALUDE_point_on_line_equidistant_from_axes_in_first_quadrant_l3635_363530

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

-- Define the condition for a point being equidistant from coordinate axes
def equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

-- Define the condition for a point being in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem point_on_line_equidistant_from_axes_in_first_quadrant :
  ∃ (x y : ℝ), line_equation x y ∧ equidistant_from_axes x y ∧ in_first_quadrant x y ∧
  (∀ (x' y' : ℝ), line_equation x' y' ∧ equidistant_from_axes x' y' → in_first_quadrant x' y') :=
sorry

end NUMINAMATH_CALUDE_point_on_line_equidistant_from_axes_in_first_quadrant_l3635_363530


namespace NUMINAMATH_CALUDE_alternate_color_probability_l3635_363594

/-- The probability of drawing BWBW from a box with 5 white and 6 black balls -/
theorem alternate_color_probability :
  let initial_white : ℕ := 5
  let initial_black : ℕ := 6
  let total_balls : ℕ := initial_white + initial_black
  let prob_first_black : ℚ := initial_black / total_balls
  let prob_second_white : ℚ := initial_white / (total_balls - 1)
  let prob_third_black : ℚ := (initial_black - 1) / (total_balls - 2)
  let prob_fourth_white : ℚ := (initial_white - 1) / (total_balls - 3)
  prob_first_black * prob_second_white * prob_third_black * prob_fourth_white = 2 / 33 :=
by sorry

end NUMINAMATH_CALUDE_alternate_color_probability_l3635_363594


namespace NUMINAMATH_CALUDE_overtime_pay_rate_ratio_l3635_363562

/-- Proves that the ratio of overtime pay rate to regular pay rate is 2:1 given specific conditions -/
theorem overtime_pay_rate_ratio (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) (overtime_hours : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 180 →
  overtime_hours = 10 →
  (total_pay - regular_rate * regular_hours) / overtime_hours / regular_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_overtime_pay_rate_ratio_l3635_363562


namespace NUMINAMATH_CALUDE_problem1_l3635_363561

theorem problem1 (a b : ℝ) : 3 * (a^2 - a*b) - 5 * (a*b + 2*a^2 - 1) = -7*a^2 - 8*a*b + 5 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l3635_363561


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l3635_363507

/-- A geometric sequence with a₂ = 4 and a₄ = 8 has a₆ = 16 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a2 : a 2 = 4) (h_a4 : a 4 = 8) : a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l3635_363507


namespace NUMINAMATH_CALUDE_water_percentage_in_dried_grapes_l3635_363552

/-- Given that fresh grapes contain 60% water by weight and 30 kg of fresh grapes
    yields 15 kg of dried grapes, prove that the percentage of water in dried grapes is 20%. -/
theorem water_percentage_in_dried_grapes :
  let fresh_grape_weight : ℝ := 30
  let dried_grape_weight : ℝ := 15
  let fresh_water_percentage : ℝ := 60
  let water_weight_fresh : ℝ := fresh_grape_weight * (fresh_water_percentage / 100)
  let solid_weight : ℝ := fresh_grape_weight - water_weight_fresh
  let water_weight_dried : ℝ := dried_grape_weight - solid_weight
  let dried_water_percentage : ℝ := (water_weight_dried / dried_grape_weight) * 100
  dried_water_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_water_percentage_in_dried_grapes_l3635_363552


namespace NUMINAMATH_CALUDE_find_e_l3635_363527

/-- Given two functions f and g, and a composition condition, prove the value of e. -/
theorem find_e (b e : ℝ) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * x + b)
  (g : ℝ → ℝ) (hg : ∀ x, g x = b * x + 5)
  (h_comp : ∀ x, f (g x) = 15 * x + e) : 
  e = 15 := by sorry

end NUMINAMATH_CALUDE_find_e_l3635_363527


namespace NUMINAMATH_CALUDE_translation_result_l3635_363542

/-- Represents a 2D point with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Translates a point by given x and y offsets -/
def translate (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_result :
  let initial_point : Point := { x := -5, y := 1 }
  let final_point : Point := translate (translate initial_point 2 0) 0 (-4)
  final_point = { x := -3, y := -3 } := by sorry

end NUMINAMATH_CALUDE_translation_result_l3635_363542


namespace NUMINAMATH_CALUDE_four_common_tangents_l3635_363531

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Define the number of common tangent lines
def num_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem four_common_tangents :
  num_common_tangents circle_C1 circle_C2 = 4 := by sorry

end NUMINAMATH_CALUDE_four_common_tangents_l3635_363531


namespace NUMINAMATH_CALUDE_sum_20_terms_eq_2870_l3635_363559

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n^2

-- Define the sum of the first n terms of the sequence
def sum_a (n : ℕ) : ℕ := 
  (n * (n + 1) * (2 * n + 1)) / 6

-- Theorem statement
theorem sum_20_terms_eq_2870 :
  sum_a 20 = 2870 := by sorry

end NUMINAMATH_CALUDE_sum_20_terms_eq_2870_l3635_363559


namespace NUMINAMATH_CALUDE_fans_with_all_items_fans_with_all_items_is_27_l3635_363533

def total_fans : ℕ := 5000
def tshirt_interval : ℕ := 90
def cap_interval : ℕ := 45
def scarf_interval : ℕ := 60

theorem fans_with_all_items : ℕ := by
  -- The number of fans who received all three promotional items
  -- is equal to the floor division of total_fans by the LCM of
  -- tshirt_interval, cap_interval, and scarf_interval
  sorry

-- Prove that fans_with_all_items equals 27
theorem fans_with_all_items_is_27 : fans_with_all_items = 27 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_fans_with_all_items_is_27_l3635_363533


namespace NUMINAMATH_CALUDE_complex_magnitude_sqrt_5_l3635_363532

def complex (a b : ℝ) := a + b * Complex.I

theorem complex_magnitude_sqrt_5 (a b : ℝ) (h : a / (1 - Complex.I) = 1 - b * Complex.I) :
  Complex.abs (complex a b) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sqrt_5_l3635_363532


namespace NUMINAMATH_CALUDE_simplify_expression_l3635_363536

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  (a/b + b/a)^2 - 1/(a^2*b^2) = 2/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3635_363536


namespace NUMINAMATH_CALUDE_square_area_proof_l3635_363589

theorem square_area_proof (x : ℝ) :
  (5 * x - 18 = 25 - 2 * x) →
  (5 * x - 18 ≥ 0) →
  ((5 * x - 18)^2 : ℝ) = 7921 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3635_363589


namespace NUMINAMATH_CALUDE_largest_angle_in_specific_triangle_l3635_363501

/-- The largest angle in a triangle with sides 3√2, 6, and 3√10 is 135° --/
theorem largest_angle_in_specific_triangle : 
  ∀ (a b c : ℝ) (θ : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 6 →
  c = 3 * Real.sqrt 10 →
  c > a ∧ c > b →
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
  θ = 135 * (π / 180) := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_specific_triangle_l3635_363501


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3635_363526

theorem no_solution_for_equation : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (1 / x + 1 / y = 3 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3635_363526


namespace NUMINAMATH_CALUDE_revenue_increase_when_doubled_l3635_363512

/-- Production function model -/
noncomputable def Q (A K L α₁ α₂ : ℝ) : ℝ := A * K^α₁ * L^α₂

/-- Theorem: When α₁ + α₂ > 1, doubling inputs more than doubles revenue -/
theorem revenue_increase_when_doubled
  (A K L α₁ α₂ : ℝ)
  (h_A : A > 0)
  (h_α₁ : 0 < α₁ ∧ α₁ < 1)
  (h_α₂ : 0 < α₂ ∧ α₂ < 1)
  (h_sum : α₁ + α₂ > 1) :
  Q A (2 * K) (2 * L) α₁ α₂ > 2 * Q A K L α₁ α₂ :=
sorry

end NUMINAMATH_CALUDE_revenue_increase_when_doubled_l3635_363512
